# PDF LLM Sorter

OCRとローカルLLM（Ollama）を活用して、PDFや画像ファイルを自動解析・分類・リネームして整理するCLIツールです。

---

## 🌟 主な特徴

- **ハイブリッドテキスト抽出**:
  - デジタルPDFは [PyMuPDF](https://github.com/pymupdf/PyMuPDF) で高速にテキストレイヤーを直接抽出。
  - スキャンPDFや画像ファイル（PNG, JPG, WebP等）は、OllamaのVision/OCRモデル（例: `deepseek-ocr`, `llava` 等）を呼び出して自動文字起こし。
- **インテリジェントな分類・リネーム**:
  - ドキュメントの発行日・発行元・書類種別をLLMが文脈から把握し、統一感のあるファイル名（例: `20260401_株式会社〇〇_請求書.pdf`）へ自動リネーム。
  - 設定されたカテゴリー一覧から最適なフォルダを自動選択して配置。
- **Polars による高速なメタデータ出力**:
  - 分類結果や要約、抽出タグなどを CSV / TSV 形式で自動保存。
- **リッチでモダンなCLIインターフェース**:
  - [Typer](https://typer.tiangolo.com/) と [Rich](https://github.com/Textualize/rich) を統合し、見やすいヘルプ画面・設定パネル・カラーログ・処理結果サマリーテーブルを出力。
- **安全設計**:
  - 移動モード（`--move`）とコピーモード（`--copy`、デフォルト）を切り替え可能。
  - 同名ファイルが存在する場合は自動で連番（`_1`, `_2`）を付与し、上書きを防止。
  - 実際のファイル操作を行わずに結果を確認できる `--dry-run` 対応。

---

## 📋 必要要件

- **Python**: 3.14 以上
- **パッケージマネージャー**: [uv](https://github.com/astral-sh/uv) 推奨
- **Ollama**: ローカルまたはリモートで稼働中の Ollama サーバー
  - 推奨チャットモデル: `qwen3.5:latest` など
  - 推奨OCRモデル: `deepseek-ocr:latest` など（画像・スキャンPDFを扱う場合）

---

## 🚀 セットアップ

### 1. リポジトリのクローン & 依存関係の同期

```bash
git clone https://github.com/your-username/pdf-llm-sorter.git
cd pdf-llm-sorter

# uv を使用して仮想環境の構築と依存パッケージをインストール
uv sync
```

### 2. 設定ファイルの作成

設定のテンプレート `pdf_llm_sorter/example.config.toml` をコピーして `config.toml` を作成します。

```bash
cp pdf_llm_sorter/example.config.toml pdf_llm_sorter/config.toml
```

必要に応じて `config.toml` 内の Ollama URL、モデル名、カテゴリー一覧、フォルダパスなどを編集してください。

---

## ⚙️ 設定ファイル (`config.toml`)

```toml
[ollama]
base_url = "http://localhost:11434"
ocr_model = "deepseek-ocr:latest"    # スキャン文書・画像用OCRモデル
chat_model = "qwen3.5:latest"        # ドキュメント分類・リネーム用LLM

[prompt]
system_prompt = """あなたはPDFドキュメントの整理・分類を専門とするAIアシスタントです。
...
### カテゴリー
{{categories}}
"""

[file_system]
input_folder = "./input"             # 整理対象ファイルが格納されたフォルダ
output_folder = "./output"           # 分類先ルートフォルダ
action_mode = "copy"                 # "copy"（複製） または "move"（移動）
export_format = "csv"                # "csv", "tsv", "both", "none"
export_path = "./log"                # ログ保存先（空文字の場合は output_folder 直下）
export_with_timestamp = true         # ログファイル名にタイムスタンプを付与
recursive = false                    # サブディレクトリを再帰的に走査するかどうか

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
| `inputs...` | 処理対象のPDF/画像ファイルまたはディレクトリパス（指定時は input_folder より優先） |
| `-c, --config PATH` | 設定ファイル（TOML）のパスを指定 |
| `-o, --output PATH` | 出力先フォルダパス（設定ファイルの `output_folder` を上書き） |
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

#### 3. 元ファイルを移動して整理し、再帰的に探索する
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