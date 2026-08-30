"""Ollama を用いたドキュメント解析・分類・構造化出力モジュール。

LangChain の ChatOllama と Pydantic モデル（FileModel）を連携し、
OCR 抽出テキストからファイル名と配置先ディレクトリを自動決定します。
"""

import json
import logging
import re
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from pdf_llm_sorter.libs.config import AppConfig, OllamaConfig, PromptConfig
from pdf_llm_sorter.libs.model import FileModel

logger = logging.getLogger("pdf_llm_sorter.chat")

T = TypeVar("T", bound=BaseModel)

DEFAULT_SYSTEM_PROMPT = """あなたはPDFドキュメントの整理・分類を専門とするAIアシスタントです。
提供されたドキュメントのテキスト内容を詳細に解析し、ファイルの適切な配置先カテゴリと分かりやすいリネーム後ファイル名を決定してください。

### ルール:
1. **file_name**:
   - 書類の内容が一目でわかる具体的な名前にしてください。
   - 推奨フォーマット: `[日付]_[発行元または組織名]_[書類の種類・件名].pdf`
   - 日付が特定できる場合は先頭に `YYYYMMDD` を付与してください。
   - 必ず末尾に `.pdf` を含めてください。
2. **category**:
   - 設定されたカテゴリー一覧の中から最も適したものを正確に1つ選択してください。
3. **document_date**:
   - 書類に記載された発行日・作成日・契約日など（YYYY-MM-DD形式、不明なら空文字）。
4. **issuer**:
   - 書類の発行元企業名・団体名・送信者。
5. **summary**:
   - 書類の重要事項（金額、期日、対象者、内容）の1〜2行程度の簡潔な要約。
6. **tags**:
   - 検索に役立つキーワードタグ（3〜5個程度）。
7. **表記・誤字補正**:
   - OCRの読み取り誤り（例: カタカナのハ/ノの誤認、タウンノウジング→タウンハウジングなど）がある場合は、文脈や一般的な組織名・日本語の自然な表記に合わせて自動補正してください。

### カテゴリー:
{{categories}}
"""


def format_categories_for_prompt(categories: list[str] | dict[str, str] | None) -> str:
    """カテゴリリストまたは辞書をプロンプト埋め込み用の箇条書き文字列に変換します。"""
    if not categories:
        return "- その他"
    if isinstance(categories, dict):
        return "\n".join(f"- **{cat}**: {desc}" for cat, desc in categories.items())
    return "\n".join(f"- {cat}" for cat in categories)


def render_system_prompt(
    raw_prompt: str, categories: list[str] | dict[str, str] | None
) -> str:
    """システムプロンプト内の {{categories}} プレースホルダーをカテゴリ一覧で置換します。"""
    prompt = raw_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    cat_str = format_categories_for_prompt(categories)

    if "{{categories}}" in prompt:
        return prompt.replace("{{categories}}", cat_str)
    elif categories:
        # プレースホルダーが含まれていない場合は末尾に追加
        return f"{prompt}\n\n### カテゴリー候補:\n{cat_str}"
    return prompt


def clean_json_markdown(text: str) -> str:
    """LLM 出力からマークダウンのコードブロックを除去して生の JSON 文字列を抽出します。"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def truncate_document_text(text: str, max_chars: int = 6000) -> str:
    """長文テキストを先頭と末尾の重要情報を保持しつつ最大文字数内に切り詰めます。

    発行日・発行元・書類タイトルがある先頭部（約70%）と、
    署名・捺印・合計金額等がある末尾部（約30%）を残し、中間を省略します。

    Args:
        text: 元のドキュメントテキスト
        max_chars: 最大文字数（0以下の場合は切り詰めなし）

    Returns:
        str: 切り詰め後のテキスト
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    omission_msg = "\n\n... [中略: ドキュメント中間部分を省略] ...\n\n"
    available_chars = max(0, max_chars - len(omission_msg))

    # 先頭70%、末尾30%
    head_chars = int(available_chars * 0.7)
    tail_chars = available_chars - head_chars

    head_part = text[:head_chars].rstrip()
    tail_part = text[-tail_chars:].lstrip() if tail_chars > 0 else ""

    truncated = f"{head_part}{omission_msg}{tail_part}"
    logger.info(
        "ドキュメントテキストを最大文字数制限 (%d 文字) に基づき切り詰めました: %d 文字 -> %d 文字",
        max_chars,
        len(text),
        len(truncated),
    )
    return truncated


class OllamaChatClassifier:
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

        # JSON 出力モードで初期化（明示的なタイムアウトを指定）
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
        # トークンあふれ防止のためのテキスト切り詰め
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

    def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str | None = None,
    ) -> T:
        """任意の Pydantic スキーマに基づいた構造化出力を生成します。"""
        sys_p = (
            render_system_prompt(system_prompt, self.categories)
            if system_prompt
            else self.system_prompt
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=sys_p),
            HumanMessage(content=prompt),
        ]
        structured_llm = self.llm.with_structured_output(schema)
        return structured_llm.invoke(messages)

    def chat(self, prompt_or_messages: str | list[BaseMessage]) -> str:
        """シンプルなテキスト対話を実行します。"""
        if isinstance(prompt_or_messages, str):
            messages: list[BaseMessage] = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt_or_messages),
            ]
        else:
            messages = prompt_or_messages

        response = self.llm.invoke(messages)
        return str(response.content)
