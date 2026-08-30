"""OpenRouter および OpenAI 互換 API によるドキュメント分類モジュール。"""

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
from pdf_llm_sorter.libs.config import AppConfig, OpenRouterConfig, PromptConfig
from pdf_llm_sorter.libs.model import FileModel

logger = logging.getLogger("pdf_llm_sorter.chat.openrouter")


class OpenRouterChatClassifier(BaseChatClassifier):
    """OpenRouter (および OpenAI 互換 API) によるドキュメント分類クラス"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen3.7-flash",
        system_prompt: str = "",
        categories: list[str] | dict[str, str] | None = None,
        temperature: float = 0.0,
        max_chars_per_doc: int = 6000,
        timeout: float = 60.0,
        max_retries: int = 1,
        provider_name: str = "openrouter",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = (
            f"{self.base_url}/chat/completions"
            if not self.base_url.endswith("/chat/completions")
            else self.base_url
        )
        self.model = model
        self.categories = categories
        self.system_prompt = render_system_prompt(system_prompt, self.categories)
        self.temperature = temperature
        self.max_chars_per_doc = max_chars_per_doc
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider_name = provider_name

    @classmethod
    def from_config(
        cls,
        config: AppConfig | OpenRouterConfig,
        prompt_config: PromptConfig | None = None,
    ) -> OpenRouterChatClassifier:
        """設定オブジェクトからインスタンスを生成します。"""
        categories: list[str] | dict[str, str] = []
        max_chars = 6000
        if isinstance(config, AppConfig):
            or_cfg = config.openrouter
            sys_prompt = config.prompt.system_prompt
            max_chars = config.prompt.max_chars_per_doc
            categories = config.file_system.categories
        else:
            or_cfg = config
            sys_prompt = prompt_config.system_prompt if prompt_config else ""
            max_chars = prompt_config.max_chars_per_doc if prompt_config else 6000

        model = or_cfg.model or "qwen/qwen3.7-flash"
        return cls(
            api_key=or_cfg.get_api_key(),
            base_url=or_cfg.base_url,
            model=model,
            system_prompt=sys_prompt,
            categories=categories,
            max_chars_per_doc=max_chars,
            timeout=or_cfg.timeout,
            max_retries=or_cfg.max_retries,
            provider_name="openrouter",
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
                f"{self.provider_name.upper()} API キーが設定されていません。.env に OPENROUTER_API_KEY を設定してください。"
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
            "HTTP-Referer": "https://github.com/rmc8/pdf-llm-sorter",
            "X-Title": "PDF LLM Sorter",
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
                "%s 分類推論を開始 (試行 %d/%d, model=%s, timeout=%.1fs)...",
                self.provider_name.capitalize(),
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
                            f"{self.provider_name} API エラー ({response.status_code}): {response.text}"
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
                        "%s 分類推論に失敗しました (試行 %d/%d)。再試行します: %s",
                        self.provider_name,
                        attempt,
                        total_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "%s 分類推論が最大試行回数 (%d 回) に達しました: %s",
                        self.provider_name,
                        total_attempts,
                        e,
                    )
                    raise
