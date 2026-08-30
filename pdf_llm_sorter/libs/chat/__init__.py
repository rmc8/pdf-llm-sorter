"""チャット・ドキュメント分類モジュール。

Ollama および OpenRouter (OpenAI互換) 等の LLM を用いて
OCR抽出テキストからファイル名と配置先カテゴリを自動決定します。
"""

from pdf_llm_sorter.libs.chat.base import (
    DEFAULT_SYSTEM_PROMPT,
    BaseChatClassifier,
    clean_json_markdown,
    format_categories_for_prompt,
    render_system_prompt,
    truncate_document_text,
)
from pdf_llm_sorter.libs.chat.ollama import OllamaChatClassifier
from pdf_llm_sorter.libs.chat.openrouter import OpenRouterChatClassifier
from pdf_llm_sorter.libs.config import AppConfig


def create_chat_classifier(config: AppConfig) -> BaseChatClassifier:
    """AppConfig の chat_provider 設定に応じて適切なチャット分類器を生成します。

    Args:
        config: アプリケーション全体設定 (AppConfig)

    Returns:
        BaseChatClassifier: OpenRouterChatClassifier または OllamaChatClassifier
    """
    provider = getattr(config.general, "chat_provider", "ollama").lower()
    sys_prompt = config.prompt.system_prompt
    categories = config.file_system.categories
    max_chars = config.prompt.max_chars_per_doc

    if provider == "openrouter":
        return OpenRouterChatClassifier.from_config(config)
    elif provider == "mistral":
        # Mistral の Chat API (OpenAI互換エンドポイント)
        return OpenRouterChatClassifier(
            api_key=config.mistral.get_api_key(),
            base_url="https://api.mistral.ai/v1",
            model=config.mistral.chat_model or "mistral-small-latest",
            system_prompt=sys_prompt,
            categories=categories,
            max_chars_per_doc=max_chars,
            timeout=config.mistral.timeout,
            max_retries=config.mistral.max_retries,
            provider_name="mistral",
        )
    else:
        return OllamaChatClassifier.from_config(config)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "BaseChatClassifier",
    "OllamaChatClassifier",
    "OpenRouterChatClassifier",
    "clean_json_markdown",
    "create_chat_classifier",
    "format_categories_for_prompt",
    "render_system_prompt",
    "truncate_document_text",
]
