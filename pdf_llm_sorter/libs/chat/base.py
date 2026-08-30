"""チャット・分類モジュールの基底定義および共通ユーティリティ。"""

import logging
import re
from abc import ABC, abstractmethod

from pdf_llm_sorter.libs.model import FileModel

logger = logging.getLogger("pdf_llm_sorter.chat")

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


class BaseChatClassifier(ABC):
    """チャット分類器の抽象基底クラス"""

    provider_name: str = "llm"
    model: str = ""

    @abstractmethod
    def classify_document(
        self,
        document_text: str,
        original_filename: str = "",
        additional_instructions: str = "",
    ) -> FileModel:
        """テキストを解析して分類結果 (FileModel) を返します。"""
        pass

