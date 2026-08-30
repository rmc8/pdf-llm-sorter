"""Ollama 経由の Vision / OCR モデルを利用したドキュメント OCR クライアント。"""

import logging
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from pdf_llm_sorter.libs.config import OllamaConfig
from pdf_llm_sorter.libs.ocr.base import (
    BaseOCRClient,
    DocumentOCRResult,
    PageOCRResult,
    bytes_to_base64,
    image_to_base64,
    render_pdf_page_to_image,
    render_pdf_to_images,
)

logger = logging.getLogger("pdf_llm_sorter.ocr.ollama")

# 互換性のためのエイリアス再エクスポート
__all__ = [
    "DocumentOCRResult",
    "OllamaOCRClient",
    "PageOCRResult",
    "bytes_to_base64",
    "extract_text_from_pdf",
    "image_to_base64",
    "render_pdf_page_to_image",
    "render_pdf_to_images",
]


class OllamaOCRClient(BaseOCRClient):
    """Ollama API を利用した OCR クライアント"""

    provider_name: str = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-ocr:latest",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @classmethod
    def from_config(
        cls, config: OllamaConfig, timeout: float | None = None
    ) -> OllamaOCRClient:
        """OllamaConfig からインスタンスを生成します。"""
        model = config.ocr_model or "deepseek-ocr:latest"
        req_timeout = timeout if timeout is not None else config.timeout
        return cls(
            base_url=config.base_url,
            model=model,
            timeout=req_timeout,
            max_retries=config.max_retries,
        )

    def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """画像バイトデータから OCR を実行してテキストを抽出します。"""
        img_b64 = bytes_to_base64(image_bytes)
        return self._call_generate_api(img_b64, prompt)

    def _call_generate_api(self, image_base64: str, prompt: str) -> str:
        """Ollama の /api/generate エンドポイントを呼び出します（リトライ対応）。"""
        url = f"{self.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
        }

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            logger.debug(
                "Ollama OCR API 呼び出し (試行 %d/%d): %s (model=%s, timeout=%.1fs)",
                attempt,
                total_attempts,
                url,
                self.model,
                self.timeout,
            )

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    if response.status_code != 200:
                        error_detail = response.text
                        raise RuntimeError(
                            f"Ollama OCR API エラー ({response.status_code}): {error_detail}"
                        )

                    data = response.json()
                    extracted_text = data.get("response", "").strip()
                    return extracted_text
            except Exception as e:
                if attempt < total_attempts:
                    logger.warning(
                        "OCR 呼び出しに失敗しました (試行 %d/%d)。再試行します: %s",
                        attempt,
                        total_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "OCR 呼び出しが最大試行回数 (%d 回) に達しました: %s",
                        total_attempts,
                        e,
                    )
                    raise


def extract_text_from_pdf(
    pdf_path: Path | str,
    ocr_client: BaseOCRClient | None = None,
    force_ocr: bool = False,
    min_text_chars_per_page: int = 30,
    dpi: int = 150,
    prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    max_pages: int | None = None,
    enable_ocr: bool = True,
) -> DocumentOCRResult:
    """PDF からテキストを抽出します。

    テキストレイヤーが存在する場合は直接テキストを抽出し、
    スキャン画像などテキストが不十分な場合（または force_ocr=True の場合）は
    enable_ocr=True であれば OCR クライアント（Mistral / DeepSeek / Ollama）を使用して文字を抽出します。
    enable_ocr=False の場合は OCR をスキップし、埋め込みテキストのみを高速に取得します。

    Args:
        pdf_path: 対象のPDFファイルパス
        ocr_client: BaseOCRClient インスタンス (OCRが必要な場合に使用)
        force_ocr: True の場合、テキストレイヤーがあっても常に OCR を実行
        min_text_chars_per_page: 埋め込みテキストがこの文字数未満ならスキャンとみなしてOCRを実行
        dpi: OCR用画像レンダリング時の解像度 (DPI)
        prompt: OCR モデルへの指示プロンプト
        max_pages: 解析対象の最大ページ数（None または 0 以下の場合は全ページ）
        enable_ocr: 画像スキャンに対するOCR処理を実行するかどうか（False時は高速スキップ）

    Returns:
        DocumentOCRResult: ページごとの抽出結果および全テキスト
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDFファイルが見つかりません: {path}")

    # force_ocr でかつクライアントが PDF 直接一括解析に対応している場合（Mistral等）
    if enable_ocr and force_ocr and ocr_client is not None:
        direct_res = ocr_client.extract_from_pdf_file(path, max_pages=max_pages)
        if direct_res is not None:
            return direct_res

    doc = pymupdf.open(path)
    total_pages = len(doc)
    page_results: list[PageOCRResult] = []

    pages_to_process = (
        min(total_pages, max_pages) if max_pages and max_pages > 0 else total_pages
    )

    if pages_to_process < total_pages:
        logger.info(
            "PDF テキスト抽出開始: %s (全 %d ページ中 先頭 %d ページを抽出)",
            path.name,
            total_pages,
            pages_to_process,
        )
    else:
        logger.info("PDF テキスト抽出開始: %s (全 %d ページ)", path.name, total_pages)

    try:
        # 全体で埋め込みテキストが完全にゼロの場合かつ PDF一括対応クライアントの場合
        if enable_ocr and ocr_client is not None:
            # 試しに先頭ページの埋め込み文字数をチェック
            sample_text = "".join(
                doc[i].get_text("text").strip() for i in range(min(pages_to_process, 3))
            )
            if len(sample_text) < min_text_chars_per_page:
                direct_res = ocr_client.extract_from_pdf_file(
                    path, max_pages=pages_to_process
                )
                if direct_res is not None:
                    return direct_res

        for idx in range(pages_to_process):
            page_num = idx + 1
            page = doc[idx]
            embedded_text = page.get_text("text").strip()

            should_use_ocr = enable_ocr and (
                force_ocr or (len(embedded_text) < min_text_chars_per_page)
            )

            if not should_use_ocr:
                if embedded_text:
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
                    logger.info(
                        "ページ %d/%d: 埋め込みテキストなし (OCR スキップ設定のため空文字として処理)",
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

            provider = getattr(ocr_client, "provider_name", "ocr")
            logger.info(
                "ページ %d/%d: %s OCR (%s) による文字認識を実行中...",
                page_num,
                total_pages,
                provider.capitalize(),
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
            method_name = f"{provider}_ocr"
            if method_name not in ("mistral_ocr", "deepseek_ocr", "ollama_ocr"):
                method_name = "ollama_ocr"

            page_results.append(
                PageOCRResult(
                    page_number=page_num,
                    text=extracted_ocr_text,
                    method=method_name,  # type: ignore[arg-type]
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
