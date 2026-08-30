"""Ollama によるテキスト対話および構造化出力モジュール。"""

import json
import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from pdf_llm_sorter.libs.chat.base import (
    BaseChatClassifier,
    clean_json_markdown,
    render_system_prompt,
    truncate_document_text,
)
from pdf_llm_sorter.libs.config import AppConfig, OllamaConfig, PromptConfig
from pdf_llm_sorter.libs.model import FileModel

logger = logging.getLogger("pdf_llm_sorter.chat.ollama")


class OllamaChatClassifier(BaseChatClassifier):
    """Ollama によるテキスト対話および構造化出力を管理するクラス"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:latest",
        system_prompt: str = "",
        categories: list[str] | dict[str, str] | None = None,
        temperature: float = 0.0,
        max_chars_per_doc: int = 6000,
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.categories = categories
        self.system_prompt = render_system_prompt(system_prompt, self.categories)
        self.temperature = temperature
        self.max_chars_per_doc = max_chars_per_doc
        self.timeout = timeout
        self.max_retries = max_retries

        # JSON 出力モードで初期化
        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            format="json",
            request_timeout=self.timeout,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig | OllamaConfig,
        prompt_config: PromptConfig | None = None,
    ) -> OllamaChatClassifier:
        """設定オブジェクトからインスタンスを生成します。"""
        categories: list[str] | dict[str, str] = []
        max_chars = 6000
        if isinstance(config, AppConfig):
            ollama_cfg = config.ollama
            sys_prompt = config.prompt.system_prompt
            max_chars = config.prompt.max_chars_per_doc
            categories = config.file_system.categories
        else:
            ollama_cfg = config
            sys_prompt = prompt_config.system_prompt if prompt_config else ""
            max_chars = prompt_config.max_chars_per_doc if prompt_config else 6000

        model = ollama_cfg.chat_model or "qwen3.5:latest"
        return cls(
            base_url=ollama_cfg.base_url,
            model=model,
            system_prompt=sys_prompt,
            categories=categories,
            max_chars_per_doc=max_chars,
            timeout=ollama_cfg.timeout,
            max_retries=ollama_cfg.max_retries,
        )

    def classify_document(
        self,
        document_text: str,
        original_filename: str = "",
        additional_instructions: str = "",
    ) -> FileModel:
        """OCR抽出テキストを解析し、FileModel（リネーム名・配置先等）の構造化データを返します（リトライ対応）。"""
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
            "必ず以下のキーを持つJSONオブジェクトのみを出力してください:\n"
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

        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            logger.info(
                "Ollama 分類推論を開始 (試行 %d/%d, model=%s, 入力文字数=%d, timeout=%.1fs)...",
                attempt,
                total_attempts,
                self.model,
                len(document_text),
                self.timeout,
            )

            try:
                response = self.llm.invoke(messages)
                raw_content = str(response.content)
                json_str = clean_json_markdown(raw_content)

                try:
                    data: dict[str, Any] = json.loads(json_str)
                    result = FileModel.model_validate(data)
                except Exception as parse_err:
                    logger.warning(
                        "JSON直接パースエラーのためLangChain構造化出力にフォールバックします: %s",
                        parse_err,
                    )
                    structured_llm = self.llm.with_structured_output(FileModel)
                    result = structured_llm.invoke(messages)

                logger.info(
                    "分類完了 -> ファイル名: '%s', カテゴリ: '%s'",
                    result.file_name,
                    result.category,
                )
                return result

            except Exception as e:
                if attempt < total_attempts:
                    logger.warning(
                        "分類推論に失敗しました (試行 %d/%d)。再試行します: %s",
                        attempt,
                        total_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "分類推論が最大試行回数 (%d 回) に達しました: %s",
                        total_attempts,
                        e,
                    )
                    raise
