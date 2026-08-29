"""設定ファイル (config.toml) の読み込みおよびバリデーションを行うモジュール。"""

from pathlib import Path
import tomllib
from typing import Any

from pydantic import BaseModel, Field


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


class AppConfig(BaseModel):
    """アプリケーション全体の設定"""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)


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
        return AppConfig.model_validate(data)
    except Exception as e:
        raise ValueError(f"設定ファイル ({path}) の読み込みに失敗しました: {e}") from e
