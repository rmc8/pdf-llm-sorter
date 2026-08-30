"""チャット・ドキュメント分類モジュール。

Ollama, OpenRouter, Mistral 等の LLM を用いて
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
from pdf_llm_sorter.libs.chat.mistral import MistralChatClassifier
from pdf_llm_sorter.libs.chat.ollama import OllamaChatClassifier
from pdf_llm_sorter.libs.chat.openrouter import OpenRouterChatClassifier
from pdf_llm_sorter.libs.config import AppConfig


def create_chat_classifier(config: AppConfig) -> BaseChatClassifier:
    """AppConfig の chat_provider 設定に応じて適切なチャット分類器を生成します。

    Args:
        config: アプリケーション全体設定 (AppConfig)

    Returns:
        BaseChatClassifier: OpenRouterChatClassifier, MistralChatClassifier, または OllamaChatClassifier
    """
    provider = getattr(config.general, "chat_provider", "ollama").lower()

    if provider == "openrouter":
        return OpenRouterChatClassifier.from_config(config)
    elif provider == "mistral":
        return MistralChatClassifier.from_config(config)
    else:
        return OllamaChatClassifier.from_config(config)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "BaseChatClassifier",
    "MistralChatClassifier",
    "OllamaChatClassifier",
    "OpenRouterChatClassifier",
    "clean_json_markdown",
    "create_chat_classifier",
    "format_categories_for_prompt",
    "render_system_prompt",
    "truncate_document_text",
]
