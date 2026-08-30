"""OCR モジュール - Mistral, Ollama によるテキスト抽出・文字認識を提供します。"""

from pdf_llm_sorter.libs.config import AppConfig
from pdf_llm_sorter.libs.ocr.base import (
    BaseOCRClient,
    DocumentOCRResult,
    PageOCRResult,
    bytes_to_base64,
    image_to_base64,
    render_pdf_page_to_image,
    render_pdf_to_images,
)
from pdf_llm_sorter.libs.ocr.mistral import MistralOCRClient
from pdf_llm_sorter.libs.ocr.ollama import OllamaOCRClient, extract_text_from_pdf


def create_ocr_client(config: AppConfig) -> BaseOCRClient:
    """AppConfig の OCR 設定に基づいて適切な OCR クライアントを生成します。

    Args:
        config: アプリケーション全体設定 (AppConfig)

    Returns:
        BaseOCRClient: MistralOCRClient または OllamaOCRClient
    """
    provider = getattr(config.ocr, "provider", "mistral").lower()

    if provider == "ollama":
        return OllamaOCRClient.from_config(config.ollama)
    else:
        # デフォルトは Mistral
        return MistralOCRClient.from_config(config.mistral)


__all__ = [
    "BaseOCRClient",
    "DocumentOCRResult",
    "MistralOCRClient",
    "OllamaOCRClient",
    "PageOCRResult",
    "bytes_to_base64",
    "create_ocr_client",
    "extract_text_from_pdf",
    "image_to_base64",
    "render_pdf_page_to_image",
    "render_pdf_to_images",
]
