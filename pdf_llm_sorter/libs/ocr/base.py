"""OCR クライアント共通基底クラスおよびデータ構造モジュール。"""

import base64
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import pymupdf
from PIL import Image
from pydantic import BaseModel, Field


class PageOCRResult(BaseModel):
    """PDF各ページのテキスト抽出結果"""

    page_number: int = Field(description="1始まりのページ番号")
    text: str = Field(description="抽出されたテキスト内容")
    method: Literal["text_layer", "mistral_ocr", "ollama_ocr", "empty"] = Field(
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
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_base64(data: bytes) -> str:
    """画像バイトデータを Base64 文字列にエンコードします。"""
    return base64.b64encode(data).decode("utf-8")


def render_pdf_page_to_image(page: pymupdf.Page, dpi: int = 150) -> Image.Image:
    """PDF ページを指定 DPI の PIL Image に変換します。"""
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


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


class BaseOCRClient(ABC):
    """OCR クライアント共通基底クラス"""

    provider_name: str = "base"
    model: str = ""

    @abstractmethod
    def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """画像バイトデータから OCR を実行してテキストを抽出します。"""
        pass

    def extract_from_pil_image(
        self,
        image: Image.Image,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """PIL Image から OCR を実行してテキストを抽出します。"""
        buf = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        return self.extract_from_image_bytes(buf.getvalue(), prompt=prompt)

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

    def extract_from_pdf_file(
        self,
        pdf_path: Path | str,
        max_pages: int | None = None,
    ) -> DocumentOCRResult | None:
        """PDFファイルを直接受け取って抽出可能な場合は DocumentOCRResult を返します。

        未対応（ページ単位レンダリングが必要）な場合は None を返します。
        """
        return None
