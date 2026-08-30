# PDF LLM Sorter

[![Python 3.14+](https://img.shields.io/badge/Python-3.14+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Mistral AI](https://img.shields.io/badge/OCR-Mistral%20AI-FD5A1E.svg)](https://mistral.ai/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6366F1.svg)](https://openrouter.ai/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg?logo=ollama&logoColor=white)](https://ollama.com/)
[![Polars](https://img.shields.io/badge/DataFrame-Polars-CD792C.svg?logo=polars&logoColor=white)](https://pola.rs/)
[![Typer](https://img.shields.io/badge/CLI-Typer-2496ED.svg)](https://typer.tiangolo.com/)

OCR（Mistral / Ollama）と LLM（OpenRouter / Mistral / Ollama）を活用して、PDFファイルを自動解析・分類・リネームして整理するCLIツールです。

---

## 🌟 主な特徴

- **マルチプロバイダー対応のハイブリッドテキスト抽出**:
  - **デジタルPDF**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF) で埋め込みテキストレイヤーを高速抽出。
  - **スキャンPDF**: **Mistral OCR API** (`mistral-ocr-latest`) または ローカル **Ollama Vision** (`deepseek-ocr:latest` 等) を使用して高精度に文字起こし。
  - **OCRスキップ（高速化）**: テキストレイヤーのみ抽出してOCRを省略する `--no-ocr` オプションに対応。
- **インテリジェントな分類・リネーム**:
  - **OpenRouter** (`qwen/qwen3.7-flash` 等)、**Mistral Chat** (`mistral-small-latest`)、またはローカル **Ollama** (`yuiseki/sarashina2.2:3b` 等) を自由に選択可能。
  - ドキュメントの発行日・発行元・書類種別を文脈から把握し、統一感のあるファイル名（例: `20260401_株式会社〇〇_請求書.pdf`）へ自動リネーム。
  - 設定されたカテゴリー一覧から最適なフォルダを自動選択して配置。
- **安全な API キー管理**:
  - `config.toml` に API キーを直書きする必要はなく、`python-dotenv` により `.env` ファイル（`MISTRAL_API_KEY`, `OPENROUTER_API_KEY`）から安全に自動読み込み。
- **Polars による高速なメタデータ出力**:
  - 分類結果や要約、抽出タグなどを CSV / TSV 形式で自動保存。
- **リッチでモダンなCLIインターフェース**:
  - [Typer](https://typer.tiangolo.com/) と [Rich](https://github.com/Textualize/rich) を統合し、見やすい設定パネル、カラーログ、プログレスバー、処理結果サマリーテーブルを出力。
- **安全設計**:
  - 移動モード（`--move`）とコピーモード（`--copy`、デフォルト）を切り替え可能。
  - 同名ファイルが存在する場合は自動で連番（`_1`, `_2`）を付与し、上書きを防止。
  - 実際のファイル操作を行わずに結果を確認できるシミュレーションモード（`-n, --dry-run`）対応。

---

## 📋 必要要件

- **Python**: 3.14 以上
- **パッケージマネージャー**: [uv](https://github.com/astral-sh/uv) 推奨
- **APIキーまたはサーバー**:
  - クラウド利用時: Mistral API キー または OpenRouter API キー（`.env` に設定）
  - ローカル利用時: 稼働中の Ollama サーバー

---

## 🚀 セットアップ

### 1. リポジトリのクローン & 依存関係の同期

```bash
git clone https://github.com/your-username/pdf-llm-sorter.git
cd pdf-llm-sorter

# uv を使用して仮想環境の構築と依存パッケージをインストール
uv sync
```

### 2. 環境変数ファイル (`.env`) の作成

`.example.env` をコピーして `.env` を作成し、必要な API キーを設定します。

```bash
cp pdf_llm_sorter/.example.env pdf_llm_sorter/.env
```

```env
# pdf_llm_sorter/.env
MISTRAL_API_KEY="your_mistral_api_key"
OPENROUTER_API_KEY="your_openrouter_api_key"
```

### 3. 設定ファイル (`config.toml`) の作成

設定のテンプレート `pdf_llm_sorter/example.config.toml` をコピーして `config.toml` を作成します。

```bash
cp pdf_llm_sorter/example.config.toml pdf_llm_sorter/config.toml
```

---

## ⚙️ 設定ファイル (`config.toml`)

```toml
[general]
ocr_provider = "mistral"              # 使用するOCRプロバイダー ("mistral", "ollama")
chat_provider = "openrouter"           # 使用する推論プロバイダー ("openrouter", "mistral", "ollama")
dpi = 150                              # OCR用画像レンダリング解像度
min_text_chars_per_page = 30           # 埋め込み文字数がこの値未満の場合にOCRを実行
enable_ocr = true                      # 画像スキャンPDFのOCR処理を行うか

[mistral]
ocr_model = "mistral-ocr-latest"       # Mistral OCR モデル名
chat_model = "mistral-small-latest"    # Mistral 推論モデル名

[openrouter]
chat_model = "qwen/qwen3.7-flash"      # OpenRouter 推論モデル名

[ollama]
base_url = "http://localhost:11434"
ocr_model = "deepseek-ocr:latest"
chat_model = "yuiseki/sarashina2.2:3b"
timeout = 60.0                         # APIリクエストのタイムアウト秒数
max_retries = 1                        # 通信エラー発生時の自動リトライ回数
enable_ocr = false

[prompt]
system_prompt = """あなたはPDFドキュメントの整理・分類を専門とするAIアシスタントです。
...
### カテゴリー
{{categories}}
"""
max_chars_per_doc = 6000            # プロンプト最大文字数（超過時は先頭・末尾をサンプリング）

[file_system]
input_folder = "./input"             # 整理対象ファイルが格納されたフォルダ
output_folder = "./output"           # 分類先ルートフォルダ
action_mode = "copy"                 # "copy"（複製） または "move"（移動）
export_format = "csv"                # "csv", "tsv", "both", "none"
export_path = "./log"                # ログ保存先（空文字の場合は output_folder 直下）
export_with_timestamp = true         # ログファイル名にタイムスタンプを付与
recursive = false                    # サブディレクトリを再帰的に走査するかどうか
max_pages_per_pdf = 5                # PDF解析対象の最大ページ数（0で全ページ解析）

[file_system.categories]
"領収書" = "レシートや納品書などの支払明細や領収書"
"請求書" = "取引先やサービスからの請求書や支払案内"
"契約書" = "業務委託や賃貸などの各種契約書類や覚書"
"給与明細" = "給与や賞与の支払明細書"
"保険証券" = "生命保険や損害保険などの保険証券や契約内容通知書"
"税金関係" = "確定申告書や源泉徴収票などの税務関連書類"
"取扱説明書" = "家電や機器などの操作マニュアルや仕様書"
"その他" = "上記いずれのカテゴリーにも明確に該当しない一般的な文書"
```

---

## 💻 使い方

### 基本実行

設定ファイルの `input_folder` にあるファイルを分類・整理します。

```bash
uv run pdf-llm-sorter
```

### コマンドラインオプション

| オプション | 説明 |
| :--- | :--- |
| `inputs...` | 処理対象のPDFファイルまたはディレクトリパス（指定時は input_folder より優先） |
| `-c, --config PATH` | 設定ファイル（TOML）のパスを指定（デフォルト: `pdf_llm_sorter/config.toml`） |
| `-o, --output PATH` | 出力先フォルダパス（設定ファイルの `output_folder` を上書き） |
| `-p, --provider {mistral,ollama}` | OCR プロバイダーを指定 |
| `--chat-provider {openrouter,mistral,ollama}` | ドキュメント分類・推論プロバイダーを指定 |
| `--ocr / --no-ocr` | 画像スキャンPDFのOCR処理の有効/無効（`--no-ocr` で埋め込みテキストのみ抽出し高速化） |
| `--copy` | 元ファイルを保持して出力先にコピー（デフォルト） |
| `--move` | 元ファイルを出力先へ移動して整理 |
| `--format {csv,tsv,both,none}` | 結果メタデータの出力形式を指定 |
| `--no-timestamp` | 結果ログファイル名にタイムスタンプを付与せず固定ファイル名で保存 |
| `-r, --recursive` | 入力フォルダ配下のサブディレクトリを再帰的に探索 |
| `-n, --dry-run` | 実際のファイル移動・コピーを行わず、シミュレーション結果を表示 |
| `-v, --verbose` | デバッグ用の詳細ログを出力 |

### 実行例

#### 1. シミュレーション実行（Dry Run）
ファイルを実際に移動・コピーせずに、どのように分類・リネームされるかを確認します。
```bash
uv run pdf-llm-sorter --dry-run
```

#### 2. 特定のファイルを指定して実行
```bash
uv run pdf-llm-sorter path/to/document.pdf
```

#### 3. OCRプロバイダーと推論モデルをコマンドラインから切り替える
```bash
uv run pdf-llm-sorter -p mistral --chat-provider openrouter
```

#### 4. 高速モード（OCRをスキップしてテキストレイヤーのみで分類）
```bash
uv run pdf-llm-sorter --no-ocr
```

#### 5. 元ファイルを移動して整理し、再帰的に探索する
```bash
uv run pdf-llm-sorter --move --recursive
```

---

## 📁 出力構造例

```text
output/
├── 領収書/
│   ├── 20260401_Amazon_購入明細書.pdf
│   └── 20260415_セブンイレブン_レシート.pdf
├── 請求書/
│   └── 20260425_〇〇電力_電気料金請求書.pdf
├── 契約書/
│   └── 20260330_不動産管理_賃貸借契約書.pdf
└── log/
    └── classification_results_20260830_091500.csv
```

---

## 📄 ライセンス

[MIT License](LICENSE)