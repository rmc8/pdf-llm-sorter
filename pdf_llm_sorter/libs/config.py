"""設定ファイル (config.toml) の読み込みおよびバリデーションを行うモジュール。"""

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, BaseModel, Field


def load_env_file(env_path: Path | str | None = None) -> None:
    """指定されたパスまたは探索した .env ファイルから環境変数を読み込みます。"""
    if env_path:
        load_dotenv(dotenv_path=Path(env_path).resolve())
        return

    # パッケージ内 .env (pdf_llm_sorter/.env) またはカレントディレクトリ・上位ディレクトリの .env を探索
    pkg_env = Path(__file__).resolve().parent.parent / ".env"
    if pkg_env.exists():
        load_dotenv(dotenv_path=pkg_env)

    # 上位探索による .env 読み込み
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(dotenv_path=found)


class GeneralConfig(BaseModel):
    """全般設定"""

    ocr_provider: str = Field(
        default="mistral",
        description="使用する OCR プロバイダー ('mistral', 'ollama')",
    )
    chat_provider: str = Field(
        default="openrouter",
        description="使用するチャット・推論プロバイダー ('openrouter', 'mistral', 'ollama')",
    )


class OCRConfig(BaseModel):
    """OCR 全体設定"""

    provider: Literal["mistral", "ollama"] = Field(
        default="mistral",
        description="使用する OCR プロバイダー ('mistral', 'ollama')",
        validation_alias=AliasChoices("provider", "ocr_provider"),
    )
    enable_ocr: bool = Field(
        default=True,
        description="画像スキャンPDFに対するOCR処理を有効にするか（無効時は埋め込みテキストのみ抽出して高速化）",
    )
    dpi: int = Field(
        default=150,
        description="PDFページを画像化する際のレンダリング解像度 (DPI)",
    )
    min_text_chars_per_page: int = Field(
        default=30,
        description="埋め込みテキストがこの文字数未満の場合に画像スキャンと判定してOCRを実行する閾値",
    )


class MistralConfig(BaseModel):
    """Mistral 接続設定"""

    api_key: str = Field(
        default="",
        description="Mistral API キー (空文字の場合は環境変数 MISTRAL_API_KEY を参照)",
    )
    model: str = Field(
        default="mistral-ocr-latest",
        description="Mistral OCR モデル名 (例: mistral-ocr-latest)",
        validation_alias=AliasChoices("model", "ocr_model"),
    )
    chat_model: str = Field(
        default="mistral-small-latest",
        description="Mistral 分類・推論モデル名",
    )
    endpoint: str = Field(
        default="https://api.mistral.ai/v1/ocr",
        description="Mistral OCR API エンドポイント URL",
    )
    timeout: float = Field(
        default=60.0, description="Mistral API リクエストのタイムアウト秒数"
    )
    max_retries: int = Field(
        default=2, description="API 通信エラー・タイムアウト時の最大リトライ回数"
    )
    include_image_base64: bool = Field(
        default=False,
        description="レスポンスに抽出画像の base64 データを含めるかどうか",
    )

    def get_api_key(self) -> str:
        """設定値または環境変数から API キーを取得します。"""
        if self.api_key:
            return self.api_key
        return os.environ.get("MISTRAL_API_KEY", "")


class OpenRouterConfig(BaseModel):
    """OpenRouter 接続設定"""

    api_key: str = Field(
        default="",
        description="OpenRouter API キー (空文字の場合は環境変数 OPENROUTER_API_KEY を参照)",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API エンドポイント URL",
    )
    model: str = Field(
        default="qwen/qwen3.7-flash",
        description="OpenRouter 分類・推論モデル名",
        validation_alias=AliasChoices("model", "chat_model"),
    )
    timeout: float = Field(
        default=60.0, description="OpenRouter API リクエストのタイムアウト秒数"
    )
    max_retries: int = Field(
        default=1, description="API 通信エラー・タイムアウト時の最大リトライ回数"
    )

    def get_api_key(self) -> str:
        """設定値または環境変数から API キーを取得します。"""
        if self.api_key:
            return self.api_key
        return os.environ.get("OPENROUTER_API_KEY", "")


class OllamaConfig(BaseModel):
    """Ollama 接続設定"""

    base_url: str = Field(
        default="http://localhost:11434", description="Ollama APIのエンドポイントURL"
    )
    ocr_model: str = Field(
        default="deepseek-ocr:latest",
        description="OCR / Vision 処理に使用するモデル名",
        validation_alias=AliasChoices("ocr_model", "model"),
    )
    chat_model: str = Field(
        default="", description="テキスト分類・推論に使用するモデル名"
    )
    timeout: float = Field(
        default=60.0, description="Ollama APIリクエストのタイムアウト秒数"
    )
    max_retries: int = Field(
        default=1, description="API通信エラー・タイムアウト時の最大リトライ回数"
    )
    enable_ocr: bool = Field(
        default=True,
        description="画像スキャンPDFに対するOCR処理を有効にするか（無効時は埋め込みテキストのみ抽出して高速化）",
    )


class PromptConfig(BaseModel):
    """プロンプト設定"""

    system_prompt: str = Field(default="", description="分類用のシステムプロンプト")
    max_chars_per_doc: int = Field(
        default=6000,
        description="LLM推論プロンプトに渡す最大文字数（超過時は先頭と末尾をサンプリング）",
    )


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
    max_pages_per_pdf: int = Field(
        default=5,
        description="PDF解析対象の最大ページ数（0で全ページ解析）",
    )
    categories: list[str] | dict[str, str] = Field(
        default_factory=list,
        description="分類先フォルダのカテゴリ候補リストまたは説明付き辞書（指定された場合、LLMはこの中から選択）",
    )


class AppConfig(BaseModel):
    """アプリケーション全体の設定"""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    mistral: MistralConfig = Field(default_factory=MistralConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
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
    # .env ファイルの自動読み込み
    load_env_file()

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

        # 空文字の ocr_model/model を安全にフォールバック
        for section, default_model in [
            ("mistral", "mistral-ocr-latest"),
            ("openrouter", "qwen/qwen3.7-flash"),
            ("ollama", "deepseek-ocr:latest"),
        ]:
            if section in data and isinstance(data[section], dict):
                sec_dict = data[section]
                if "model" in sec_dict and not sec_dict["model"]:
                    sec_dict["model"] = default_model
                if "ocr_model" in sec_dict and not sec_dict["ocr_model"]:
                    sec_dict["ocr_model"] = default_model

        # general セクションの設定を反映
        if "general" in data and isinstance(data["general"], dict):
            gen = data["general"]
            prov = gen.get("ocr_provider", "").strip()
            if prov:
                if "ocr" not in data or not isinstance(data["ocr"], dict):
                    data["ocr"] = {}
                data["ocr"]["provider"] = prov
            if "dpi" in gen:
                if "ocr" not in data or not isinstance(data["ocr"], dict):
                    data["ocr"] = {}
                data["ocr"]["dpi"] = gen["dpi"]
            if "min_text_chars_per_page" in gen:
                if "ocr" not in data or not isinstance(data["ocr"], dict):
                    data["ocr"] = {}
                data["ocr"]["min_text_chars_per_page"] = gen["min_text_chars_per_page"]
            if "enable_ocr" in gen:
                if "ocr" not in data or not isinstance(data["ocr"], dict):
                    data["ocr"] = {}
                data["ocr"]["enable_ocr"] = gen["enable_ocr"]

        # ollama.enable_ocr が明示されており ocr.enable_ocr がない場合の互換性フォールバック
        if "ocr" not in data and "ollama" in data and isinstance(data["ollama"], dict):
            if "enable_ocr" in data["ollama"]:
                data["ocr"] = {"enable_ocr": data["ollama"]["enable_ocr"]}

        return AppConfig.model_validate(data)
    except Exception as e:
        raise ValueError(f"設定ファイル ({path}) の読み込みに失敗しました: {e}") from e
