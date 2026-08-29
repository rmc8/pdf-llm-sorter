"""PDFおよび画像ファイルからテキスト・文字情報を抽出するOCRモジュール。

PyMuPDF による PDF のページレンダリングおよびテキストレイヤー抽出と、
Ollama 経由の Vision / OCR モデル（例: deepseek-ocr:latest）による画像文字認識を提供します。
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, Literal

import httpx
from PIL import Image
from pydantic import BaseModel, Field
import pymupdf

from pdf_llm_sorter.libs.config import OllamaConfig

logger = logging.getLogger("pdf_llm_sorter.ocr")


class PageOCRResult(BaseModel):
    """PDF各ページのテキスト抽出結果"""

    page_number: int = Field(description="1始まりのページ番号")
    text: str = Field(description="抽出されたテキスト内容")
    method: Literal["text_layer", "deepseek_ocr", "empty"] = Field(
        description="テキスト取得方法"
    )


class DocumentOCRResult(BaseModel):
    """ドキュメント全体のテキスト抽出結果"""

    file_path: str = Field(description="対象ファイルのパス")
    total_pages: int = Field(description="総ページ数")
    full_text: str = Field(description="結合された全テキスト")
    pages: list[PageOCRResult] = Field(
        default_factory=list, description="ページごとの抽出結果"
    )


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """PIL Image を Base64 文字列にエンコードします。"""
    buf = io.BytesIO()
    # RGB 形式であることを確認
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_base64(data: bytes) -> str:
    """画像バイトデータを Base64 文字列にエンコードします。"""
    return base64.b64encode(data).decode("utf-8")


class OllamaOCRClient:
    """Ollama API を利用した OCR クライアント"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-ocr:latest",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_config(
        cls, config: OllamaConfig, timeout: float = 180.0
    ) -> "OllamaOCRClient":
        """OllamaConfig からインスタンスを生成します。"""
        model = config.ocr_model or "deepseek-ocr:latest"
        return cls(base_url=config.base_url, model=model, timeout=timeout)

    def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """画像バイトデータから OCR を実行してテキストを抽出します。"""
        img_b64 = bytes_to_base64(image_bytes)
        return self._call_generate_api(img_b64, prompt)

    def extract_from_pil_image(
        self,
        image: Image.Image,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """PIL Image から OCR を実行してテキストを抽出します。"""
        img_b64 = image_to_base64(image)
        return self._call_generate_api(img_b64, prompt)

    def extract_from_image_file(
        self,
        file_path: Path | str,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """画像ファイル（PNG, JPEG 等）から OCR を実行します。"""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {path}")

        with open(path, "rb") as f:
            image_bytes = f.read()
        return self.extract_from_image_bytes(image_bytes, prompt)

    def _call_generate_api(self, image_base64: str, prompt: str) -> str:
        """Ollama の /api/generate エンドポイントを呼び出します。"""
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }

        logger.debug("Ollama OCR API 呼び出し: %s (model=%s)", url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            if response.status_code != 200:
                error_detail = response.text
                logger.error(
                    "Ollama API エラー (ステータス %d): %s",
                    response.status_code,
                    error_detail,
                )
                raise RuntimeError(
                    f"Ollama OCR API エラー ({response.status_code}): {error_detail}"
                )

            data = response.json()
            extracted_text = data.get("response", "").strip()
            return extracted_text


def render_pdf_page_to_image(page: pymupdf.Page, dpi: int = 150) -> Image.Image:
    """PDF ページを指定 DPI の PIL Image に変換します。"""
    # 72 DPI がデフォルト。dpi/72 で拡大縮小マトリクスを作成
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def render_pdf_to_images(pdf_path: Path | str, dpi: int = 150) -> list[Image.Image]:
    """PDF の全ページを PIL Image のリストとしてレンダリングします。"""
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDFファイルが見つかりません: {path}")

    doc = pymupdf.open(path)
    images: list[Image.Image] = []
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            img = render_pdf_page_to_image(page, dpi=dpi)
            images.append(img)
    finally:
        doc.close()

    return images


def extract_text_from_pdf(
    pdf_path: Path | str,
    ocr_client: OllamaOCRClient | None = None,
    force_ocr: bool = False,
    min_text_chars_per_page: int = 30,
    dpi: int = 150,
    prompt: str = "画像内のテキストをそのまま書き起こしてください。",
) -> DocumentOCRResult:
    """PDF からテキストを抽出します。

    テキストレイヤーが存在する場合は直接テキストを抽出し、
    スキャン画像などテキストが不十分な場合（または force_ocr=True の場合）は
    Ollama OCR モデル（deepseek-ocr 等）を使用して画像から文字を抽出します。

    Args:
        pdf_path: 対象のPDFファイルパス
        ocr_client: OllamaOCRClient インスタンス (OCRが必要な場合に使用)
        force_ocr: True の場合、テキストレイヤーがあっても常に OCR を実行
        min_text_chars_per_page: 埋め込みテキストがこの文字数未満ならスキャンとみなしてOCRを実行
        dpi: OCR用画像レンダリング時の解像度 (DPI)
        prompt: OCR モデルへの指示プロンプト

    Returns:
        DocumentOCRResult: ページごとの抽出結果および全テキスト
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDFファイルが見つかりません: {path}")

    doc = pymupdf.open(path)
    total_pages = len(doc)
    page_results: list[PageOCRResult] = []

    logger.info("PDF テキスト抽出開始: %s (全 %d ページ)", path.name, total_pages)

    try:
        for idx in range(total_pages):
            page_num = idx + 1
            page = doc[idx]
            embedded_text = page.get_text("text").strip()

            should_use_ocr = force_ocr or (len(embedded_text) < min_text_chars_per_page)

            if not should_use_ocr and embedded_text:
                logger.info(
                    "ページ %d/%d: 埋め込みテキストを取得 (%d 文字)",
                    page_num,
                    total_pages,
                    len(embedded_text),
                )
                page_results.append(
                    PageOCRResult(
                        page_number=page_num,
                        text=embedded_text,
                        method="text_layer",
                    )
                )
            else:
                if ocr_client is None:
                    if embedded_text:
                        logger.warning(
                            "ページ %d/%d: OCR クライアントが未指定のため埋め込みテキスト (%d 文字) を使用します",
                            page_num,
                            total_pages,
                            len(embedded_text),
                        )
                        page_results.append(
                            PageOCRResult(
                                page_number=page_num,
                                text=embedded_text,
                                method="text_layer",
                            )
                        )
                    else:
                        logger.warning(
                            "ページ %d/%d: テキストが存在せず、OCR クライアントも指定されていないためスキップします",
                            page_num,
                            total_pages,
                        )
                        page_results.append(
                            PageOCRResult(
                                page_number=page_num,
                                text="",
                                method="empty",
                            )
                        )
                    continue

                logger.info(
                    "ページ %d/%d: Ollama OCR (%s) による文字認識を実行中...",
                    page_num,
                    total_pages,
                    ocr_client.model,
                )
                page_image = render_pdf_page_to_image(page, dpi=dpi)
                extracted_ocr_text = ocr_client.extract_from_pil_image(
                    page_image, prompt=prompt
                )
                logger.info(
                    "ページ %d/%d: OCR 抽出完了 (%d 文字)",
                    page_num,
                    total_pages,
                    len(extracted_ocr_text),
                )
                page_results.append(
                    PageOCRResult(
                        page_number=page_num,
                        text=extracted_ocr_text,
                        method="deepseek_ocr",
                    )
                )
    finally:
        doc.close()

    # 全ページのテキストを結合
    page_texts: list[str] = []
    for res in page_results:
        if res.text:
            page_texts.append(f"--- [Page {res.page_number}] ---\n{res.text}")

    full_text = "\n\n".join(page_texts)

    return DocumentOCRResult(
        file_path=str(path),
        total_pages=total_pages,
        full_text=full_text,
        pages=page_results,
    )
