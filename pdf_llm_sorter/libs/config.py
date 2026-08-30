"""設定ファイル (config.toml) の読み込みおよびバリデーションを行うモジュール。"""

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


class OllamaConfig(BaseModel):
    """Ollama 接続設定"""

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama APIのエンドポイントURL"
    )
    ocr_model: str = Field(
        default="", description="OCR / Vision 処理に使用するモデル名"
    )
    chat_model: str = Field(
        default="", description="テキスト分類・推論に使用するモデル名"
    )


class PromptConfig(BaseModel):
    """プロンプト設定"""

    system_prompt: str = Field(default="", description="分類用のシステムプロンプト")


class FileSystemConfig(BaseModel):
    """ファイル入出力・フォルダ設定"""

    input_folder: str = Field(
        default="./input",
        description="処理対象のPDF/画像ファイルが配置される入力フォルダ",
    )
    output_folder: str = Field(
        default="./output",
        description="分類・リネーム後のPDFファイルを配置する出力フォルダ",
        validation_alias=AliasChoices("output_folder", "ouput_folder"),
    )
    action_mode: Literal["copy", "move"] = Field(
        default="copy",
        description="ファイルの配置方法 ('copy' で元ファイルを残す、'move' で移動整理)",
    )
    export_format: Literal["csv", "tsv", "both", "none"] = Field(
        default="csv",
        description="メタデータ一覧の出力形式 ('csv', 'tsv', 'both', 'none')",
    )
    export_path: str = Field(
        default="",
        description="メタデータ一覧の出力先ファイル/フォルダパス (空文字の場合は output_folder 直下に出力。{timestamp}, {date} プレースホルダー利用可能)",
    )
    export_with_timestamp: bool = Field(
        default=True,
        description="出力ファイル名に実行タイムスタンプ (例: classification_results_20260830_090928.csv) を付与するかどうか",
    )
    recursive: bool = Field(
        default=False,
        description="input_folder 配下のサブディレクトリまで再帰的に走査するかどうか",
    )
    categories: list[str] | dict[str, str] = Field(
        default_factory=list,
        description="分類先フォルダのカテゴリ候補リストまたは説明付き辞書（指定された場合、LLMはこの中から選択）",
    )


class AppConfig(BaseModel):
    """アプリケーション全体の設定"""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    file_system: FileSystemConfig = Field(default_factory=FileSystemConfig)


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """TOML 設定ファイルを読み込み、AppConfig インスタンスを返します。

    Args:
        config_path: 設定ファイルのパス。未指定の場合はパッケージ内の config.toml を参照します。

    Returns:
        AppConfig: バリデーション済みの設定オブジェクト

    Raises:
        FileNotFoundError: 指定された設定ファイルが存在しない場合
        ValueError: TOMLのパースやバリデーションに失敗した場合
    """
    if config_path is None:
        # パッケージルート (pdf_llm_sorter/) 直下の config.toml
        default_path = Path(__file__).resolve().parent.parent / "config.toml"
        if default_path.exists():
            config_path = default_path
        else:
            config_path = Path("config.toml")

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    try:
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        # スペルミス (ouput_folder) が含まれている場合の互換性フォールバック
        if "file_system" in data and isinstance(data["file_system"], dict):
            fs = data["file_system"]
            if "ouput_folder" in fs and "output_folder" not in fs:
                fs["output_folder"] = fs["ouput_folder"]

        return AppConfig.model_validate(data)
    except Exception as e:
        raise ValueError(f"設定ファイル ({path}) の読み込みに失敗しました: {e}") from e
