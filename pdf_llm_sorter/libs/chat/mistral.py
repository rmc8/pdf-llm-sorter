"""Mistral Chat API によるドキュメント分類モジュール。"""

import json
import logging
from typing import Any

import httpx

from pdf_llm_sorter.libs.chat.base import (
    BaseChatClassifier,
    clean_json_markdown,
    render_system_prompt,
    truncate_document_text,
)
from pdf_llm_sorter.libs.config import AppConfig, MistralConfig, PromptConfig
from pdf_llm_sorter.libs.model import FileModel

logger = logging.getLogger("pdf_llm_sorter.chat.mistral")


class MistralChatClassifier(BaseChatClassifier):
    """Mistral Chat API によるドキュメント分類クラス"""

    provider_name: str = "mistral"

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "https://api.mistral.ai/v1/chat/completions",
        model: str = "mistral-small-latest",
        system_prompt: str = "",
        categories: list[str] | dict[str, str] | None = None,
        temperature: float = 0.0,
        max_chars_per_doc: int = 6000,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.categories = categories
        self.system_prompt = render_system_prompt(system_prompt, self.categories)
        self.temperature = temperature
        self.max_chars_per_doc = max_chars_per_doc
        self.timeout = timeout
        self.max_retries = max_retries

    @classmethod
    def from_config(
        cls,
        config: AppConfig | MistralConfig,
        prompt_config: PromptConfig | None = None,
    ) -> MistralChatClassifier:
        """設定オブジェクトからインスタンスを生成します。"""
        categories: list[str] | dict[str, str] = []
        max_chars = 6000
        if isinstance(config, AppConfig):
            mis_cfg = config.mistral
            sys_prompt = config.prompt.system_prompt
            max_chars = config.prompt.max_chars_per_doc
            categories = config.file_system.categories
        else:
            mis_cfg = config
            sys_prompt = prompt_config.system_prompt if prompt_config else ""
            max_chars = prompt_config.max_chars_per_doc if prompt_config else 6000

        model = mis_cfg.chat_model or "mistral-small-latest"
        return cls(
            api_key=mis_cfg.get_api_key(),
            endpoint="https://api.mistral.ai/v1/chat/completions",
            model=model,
            system_prompt=sys_prompt,
            categories=categories,
            max_chars_per_doc=max_chars,
            timeout=mis_cfg.timeout,
            max_retries=mis_cfg.max_retries,
        )

    def classify_document(
        self,
        document_text: str,
        original_filename: str = "",
        additional_instructions: str = "",
    ) -> FileModel:
        """テキストを解析して分類結果 (FileModel) を返します。"""
        api_key = self.api_key
        if not api_key:
            raise ValueError(
                "Mistral API キーが設定されていません。.env に MISTRAL_API_KEY を設定してください。"
            )

        safe_text = truncate_document_text(
            document_text, max_chars=self.max_chars_per_doc
        )

        prompt_parts: list[str] = []
        if original_filename:
            prompt_parts.append(f"### 元のファイル名:\n{original_filename}\n")

        prompt_parts.append(f"### ドキュメント本文:\n{safe_text}\n")

        if additional_instructions:
            prompt_parts.append(f"### 補足指示:\n{additional_instructions}\n")

        prompt_parts.append(
            "必ず以下のキーを持つ有効なJSONオブジェクトのみを出力してください（Markdownコードブロックは含めないでください）:\n"
            "{\n"
            '  "file_name": "リネーム後ファイル名.pdf",\n'
            '  "category": "カテゴリー名",\n'
            '  "document_date": "YYYY-MM-DD",\n'
            '  "issuer": "発行元組織名",\n'
            '  "summary": "要約",\n'
            '  "tags": ["タグ1", "タグ2"]\n'
            "}"
        )

        user_content = "\n".join(prompt_parts)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            logger.info(
                "Mistral 分類推論を開始 (試行 %d/%d, model=%s, timeout=%.1fs)...",
                attempt,
                total_attempts,
                self.model,
                self.timeout,
            )

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Mistral API エラー ({response.status_code}): {response.text}"
                        )

                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError(
                            f"レスポンスに choices が含まれていません: {data}"
                        )

                    content = choices[0]["message"]["content"]
                    json_str = clean_json_markdown(content)
                    parsed_dict = json.loads(json_str)
                    result = FileModel.model_validate(parsed_dict)

                    logger.info(
                        "分類完了 -> ファイル名: '%s', カテゴリ: '%s'",
                        result.file_name,
                        result.category,
                    )
                    return result
            except Exception as e:
                if attempt < total_attempts:
                    logger.warning(
                        "Mistral 分類推論に失敗しました (試行 %d/%d)。再試行します: %s",
                        attempt,
                        total_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "Mistral 分類推論が最大試行回数 (%d 回) に達しました: %s",
                        total_attempts,
                        e,
                    )
                    raise
