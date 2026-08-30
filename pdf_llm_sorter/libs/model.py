"""PDFドキュメントの分類・リネーム・配置先決定用 Pydantic モデル定義モジュール。"""

import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class FileModel(BaseModel):
    """LLM による PDF ファイルの解析・分類・リネーム結果モデル"""

    file_name: str = Field(
        description="リネーム後の安全なファイル名（例: 2026-08-30_株式会社ABC_請求書.pdf）。"
        "日付(YYYY-MM-DD)_発行元/件名_書類種別のフォーマットを推奨。"
    )
    category: str = Field(
        description="ファイルの配置先カテゴリ名またはサブディレクトリパス"
        "（例: 請求書・領収書, 契約書・住まい, 税金・公的書類, 給与・明細, その他）。"
    )
    document_date: str = Field(
        default="",
        description="書類に記載されている日付（YYYY-MM-DD形式、記載がない・不明な場合は空文字）。",
    )
    issuer: str = Field(
        default="",
        description="書類の発行元・会社名・組織名・送信者（例: 株式会社タウンハウジング）。",
    )
    summary: str = Field(
        default="",
        description="書類の内容の簡潔な要約（1〜2行程度）。",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="検索や整理に役立つキーワード・タグのリスト。",
    )

    @field_validator("file_name")
    @classmethod
    def sanitize_file_name(cls, v: str) -> str:
        """ファイル名から不正な文字（OS禁止文字）を除去・置換し、.pdf拡張子を保証します。"""
        cleaned = v.strip()
        # OS禁止文字をアンダースコアまたはハイフンに置換
        cleaned = re.sub(r'[\\/*?:"<>|]', "_", cleaned)
        # 連続する空白やアンダースコアを整理
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")

        if not cleaned:
            cleaned = "document"

        if not cleaned.lower().endswith(".pdf"):
            cleaned += ".pdf"

        return cleaned

    @field_validator("category")
    @classmethod
    def sanitize_category(cls, v: str) -> str:
        """カテゴリ名・パスのサニタイズを行います。"""
        cleaned = v.strip().strip("/\\")
        # 不正な記号の置換
        cleaned = re.sub(r'[*?:"<>|]', "_", cleaned)
        return cleaned or "その他"

    def get_destination_relative_path(self) -> Path:
        """分類カテゴリとファイル名を組み合わせた相対パスを返します。"""
        return Path(self.category) / self.file_name


class BatchFileClassification(BaseModel):
    """複数ファイル処理時のバッチ分類モデル"""

    results: list[FileModel] = Field(
        default_factory=list, description="分類・リネーム結果のリスト"
    )
