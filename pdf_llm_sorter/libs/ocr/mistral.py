"""Mistral OCR API (mistral-ocr-latest) を利用したドキュメント OCR クライアント。"""

import logging
from pathlib import Path
from typing import Any

import httpx

from pdf_llm_sorter.libs.config import MistralConfig
from pdf_llm_sorter.libs.ocr.base import (
    BaseOCRClient,
    DocumentOCRResult,
    PageOCRResult,
    bytes_to_base64,
)

logger = logging.getLogger("pdf_llm_sorter.ocr.mistral")


class MistralOCRClient(BaseOCRClient):
    """Mistral OCR API クライアント"""

    provider_name: str = "mistral"

    def __init__(
        self,
        api_key: str = "",
        model: str = "mistral-ocr-latest",
        endpoint: str = "https://api.mistral.ai/v1/ocr",
        timeout: float = 60.0,
        max_retries: int = 2,
        include_image_base64: bool = False,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.include_image_base64 = include_image_base64

    @classmethod
    def from_config(
        cls, config: MistralConfig, timeout: float | None = None
    ) -> MistralOCRClient:
        """MistralConfig からインスタンスを生成します。"""
        req_timeout = timeout if timeout is not None else config.timeout
        return cls(
            api_key=config.get_api_key(),
            model=config.model or "mistral-ocr-latest",
            endpoint=config.endpoint or "https://api.mistral.ai/v1/ocr",
            timeout=req_timeout,
            max_retries=config.max_retries,
            include_image_base64=config.include_image_base64,
        )

    def _get_headers(self) -> dict[str, str]:
        """API リクエスト用の HTTP ヘッダーを取得します。"""
        if not self.api_key:
            raise ValueError(
                "Mistral API キーが設定されていません。config.toml の [mistral.api_key] または環境変数 MISTRAL_API_KEY を設定してください。"
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_ocr_api(self, document_payload: dict[str, Any]) -> dict[str, Any]:
        """Mistral OCR API を呼び出します（リトライ対応）。"""
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "model": self.model,
            "document": document_payload,
            "include_image_base64": self.include_image_base64,
        }

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            logger.debug(
                "Mistral OCR API 呼び出し (試行 %d/%d): %s (model=%s, timeout=%.1fs)",
                attempt,
                total_attempts,
                self.endpoint,
                self.model,
                self.timeout,
            )

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    if response.status_code != 200:
                        error_detail = response.text
                        raise RuntimeError(
                            f"Mistral OCR API エラー ({response.status_code}): {error_detail}"
                        )

                    data: dict[str, Any] = response.json()
                    return data
            except Exception as e:
                if attempt < total_attempts:
                    logger.warning(
                        "Mistral OCR 呼び出しに失敗しました (試行 %d/%d)。再試行します: %s",
                        attempt,
                        total_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "Mistral OCR 呼び出しが最大試行回数 (%d 回) に達しました: %s",
                        total_attempts,
                        e,
                    )
                    raise

    def extract_from_image_bytes(
        self,
        image_bytes: bytes,
        prompt: str = "画像内のテキストをそのまま書き起こしてください。",
    ) -> str:
        """画像バイトデータから OCR を実行してテキストを抽出します。"""
        img_b64 = bytes_to_base64(image_bytes)
        data_uri = f"data:image/png;base64,{img_b64}"
        doc_payload = {
            "type": "image_url",
            "image_url": data_uri,
        }

        resp = self._call_ocr_api(doc_payload)
        pages = resp.get("pages", [])
        extracted_parts: list[str] = []
        for p in pages:
            markdown = p.get("markdown", "").strip()
            if markdown:
                extracted_parts.append(markdown)

        return "\n\n".join(extracted_parts)

    def extract_from_pdf_file(
        self,
        pdf_path: Path | str,
        max_pages: int | None = None,
    ) -> DocumentOCRResult | None:
        """PDF ファイルを Mistral OCR API に一括送信して DocumentOCRResult を返します。"""
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDFファイルが見つかりません: {path}")

        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()

            pdf_b64 = bytes_to_base64(pdf_bytes)
            data_uri = f"data:application/pdf;base64,{pdf_b64}"
            doc_payload = {
                "type": "document_url",
                "document_url": data_uri,
            }

            resp = self._call_ocr_api(doc_payload)
            pages_data = resp.get("pages", [])
            total_pages = len(pages_data)

            if max_pages and max_pages > 0:
                pages_data = pages_data[:max_pages]

            page_results: list[PageOCRResult] = []
            for p in pages_data:
                idx = p.get("index", len(page_results))
                md = p.get("markdown", "").strip()
                page_results.append(
                    PageOCRResult(
                        page_number=idx + 1,
                        text=md,
                        method="mistral_ocr",
                    )
                )

            page_texts = [
                f"--- [Page {r.page_number}] ---\n{r.text}"
                for r in page_results
                if r.text
            ]
            full_text = "\n\n".join(page_texts)

            return DocumentOCRResult(
                file_path=str(path),
                total_pages=total_pages,
                full_text=full_text,
                pages=page_results,
            )
        except Exception as e:
            logger.warning(
                "Mistral OCR の PDF 直接送信に失敗したため、ページ別レンダリングにフォールバックします: %s",
                e,
            )
            return None
