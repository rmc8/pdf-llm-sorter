# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-30

### Added
- **ドキュメント解析 & 自動分類パイプライン**:
  - PDF および各種画像ファイル（PNG, JPG, JPEG, WebP, TIFF, BMP）の走査・テキスト抽出・AI分類・配置処理
  - PyMuPDF を利用した高速なデジタル PDF テキスト抽出
  - Ollama Vision/OCR モデル（`deepseek-ocr` 等）を用いた画像・スキャンPDFの自動テキスト抽出
  - Ollama Chat モデル（`qwen3.5` 等）と Pydantic スキーマ（`FileModel`）による構造化出力
  - 書類の発行日・発行元・書類種別を考慮したインテリジェントなファイルリネームとフォルダ自動分類
- **メタデータログ出力**:
  - Polars を利用した分類メタデータの高速 CSV / TSV 出力
  - 実行日時タイムスタンプ付きログ保存機能
- **CLI & ユーザーインターフェース**:
  - Typer を採用した型安全で直感的なコマンドラインインターフェース
  - Rich との統合によるカラーログ出力、起動時設定パネル、処理結果サマリーテーブルの美しいターミナル表示
  - Rich `Progress` によるリアルタイムな進捗バー（件数・パーセンテージ・経過時間）および現在フェーズ（OCR・LLM推論・ファイル配置）のステータス表示
  - `pdf-llm-sorter` コマンドエントリーポイントの提供
- **安全・柔軟な実行オプション**:
  - 対象ファイルを PDF（`.pdf`）に特化し、安定したテキスト抽出と高速なドキュメント整理を実現
  - OCR スキップ（`--no-ocr` / `enable_ocr = false`）オプション: 画像スキャンに対するOCRをスキップし、埋め込みテキストレイヤーのみを抽出して全ドキュメントを爆速分類可能に
  - コピーモード（`--copy`、デフォルト）と移動モード（`--move`）の選択
  - 実際のファイル操作を行わずに結果を検証できるシミュレーションモード（`-n, --dry-run`）
  - サブディレクトリの再帰的走査（`-r, --recursive`）
  - 出力先同名ファイルとの衝突を防ぐ連番自動付与（`_1`, `_2`）ロジック
- **設定管理**:
  - TOML 形式の設定ファイル読み込み（`config.toml`）
  - 実用的なサンプルカテゴリー定義を含む `example.config.toml`
- **エラーハンドリング & 安全機構の強化**:
  - タイムアウト制御 & 自動リトライ: LLM推論およびOCR処理に明示的な `timeout`（デフォルト60秒）を設定し、通信異常やフリーズ時に最大 `max_retries`（デフォルト1回）自動再試行して安定性を大幅向上
  - 高速化チューニング: LLMパラメータを `temperature=0.0` に最適化し、最短ステップで決定論的かつ高速なJSON出力を実現
  - トークン上限・コンテキストあふれ防止: `max_chars_per_doc` による長文テキストのスマートサンプリング（先頭70%・末尾30%の重要情報保持）
  - PDF解析ページ数制限: `max_pages_per_pdf` による巨大PDFの先頭ページ抽出制御
  - 中断時ログ保護: `try ... finally` 構造により、`Ctrl+C` や例外中断時でも処理済みレコードを確実に CSV / TSV ログに書き出し
- **ドキュメント**:
  - README.md に Shields.io バッジ群（Python 3.14+, uv, Ruff, MIT, Ollama, Polars, Typer）を追加
