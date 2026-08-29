"""Ollama を用いたドキュメント解析・分類・構造化出力モジュール。

LangChain の ChatOllama と Pydantic モデル（FileModel）を連携し、
OCR 抽出テキストからファイル名と配置先ディレクトリを自動決定します。
"""

import logging
from typing import TypeVar

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
   - 日付が特定できる場合は先頭に `YYYY-MM-DD` または `YYYYMMDD` を付与してください。
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

### カテゴリー:
{{categories}}
"""


def format_categories_for_prompt(categories: list[str]) -> str:
    """カテゴリリストをプロンプト埋め込み用の箇条書き文字列に変換します。"""
    if not categories:
        return "- その他"
    return "\n".join(f"- {cat}" for cat in categories)


def render_system_prompt(raw_prompt: str, categories: list[str]) -> str:
    """システムプロンプト内の {{categories}} プレースホルダーをカテゴリ一覧で置換します。"""
    prompt = raw_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    cat_str = format_categories_for_prompt(categories)

    if "{{categories}}" in prompt:
        return prompt.replace("{{categories}}", cat_str)
    elif categories:
        # プレースホルダーが含まれていない場合は末尾に追加
        return f"{prompt}\n\n### カテゴリー候補:\n{cat_str}"
    return prompt


class OllamaChatClassifier:
    """Ollama によるテキスト対話および構造化出力を管理するクラス"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "hf.co/mmnga/sarashina2.2-3b-instruct-v0.1-gguf:latest",
        system_prompt: str = "",
        categories: list[str] | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.categories = categories or []
        self.system_prompt = render_system_prompt(system_prompt, self.categories)
        self.temperature = temperature

        self.llm = ChatOllama(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig | OllamaConfig,
        prompt_config: PromptConfig | None = None,
    ) -> "OllamaChatClassifier":
        """設定オブジェクトからインスタンスを生成します。"""
        categories: list[str] = []
        if isinstance(config, AppConfig):
            ollama_cfg = config.ollama
            sys_prompt = config.prompt.system_prompt
            categories = config.file_system.categories
        else:
            ollama_cfg = config
            sys_prompt = prompt_config.system_prompt if prompt_config else ""

        model = (
            ollama_cfg.chat_model
            or "hf.co/mmnga/sarashina2.2-3b-instruct-v0.1-gguf:latest"
        )
        return cls(
            base_url=ollama_cfg.base_url,
            model=model,
            system_prompt=sys_prompt,
            categories=categories,
        )

    def classify_document(
        self,
        document_text: str,
        original_filename: str = "",
        additional_instructions: str = "",
    ) -> FileModel:
        """OCR抽出テキストを解析し、FileModel（リネーム名・配置先等）の構造化データを返します。

        Args:
            document_text: OCRまたはテキストレイヤーから抽出したテキスト
            original_filename: 処理前の元ファイル名（参考情報）
            additional_instructions: 追加の分類指示やユーザー要望

        Returns:
            FileModel: 分類およびリネーム結果
        """
        prompt_parts: list[str] = []
        if original_filename:
            prompt_parts.append(f"### 元のファイル名:\n{original_filename}\n")

        prompt_parts.append(f"### ドキュメント本文:\n{document_text}\n")

        if additional_instructions:
            prompt_parts.append(f"### 補足指示:\n{additional_instructions}\n")

        user_content = "\n".join(prompt_parts)

        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        logger.info(
            "Ollama 分類推論を開始 (model=%s, 入力文字数=%d)...",
            self.model,
            len(document_text),
        )

        structured_llm = self.llm.with_structured_output(FileModel)
        result: FileModel = structured_llm.invoke(messages)

        logger.info(
            "分類完了 -> ファイル名: '%s', カテゴリ: '%s'",
            result.file_name,
            result.category,
        )
        return result

    def generate_structured(
        self,
        schema: type[T],
        prompt: str,
        system_prompt: str | None = None,
    ) -> T:
        """任意の Pydantic スキーマに基づいた構造化出力を生成します。

        Args:
            schema: 出力対象の Pydantic モデルクラス
            prompt: ユーザー指示プロンプト
            system_prompt: 個別のシステムプロンプト（省略時はデフォルト）

        Returns:
            指定した Pydantic スキーマのインスタンス
        """
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
