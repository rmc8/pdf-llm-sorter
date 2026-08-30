# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-30

### Added
- **マルチプロバイダー対応 OCR エンジン**:
  - **Mistral OCR**: `mistral-ocr-latest` API による高精度な PDF/画像テキスト抽出・自動リトライ処理
  - **Ollama OCR**: ローカル稼働の Vision/OCR モデル（`deepseek-ocr:latest` 等）による画像テキスト抽出
  - `BaseOCRClient` 抽象クラスおよび `create_ocr_client` ファクトリによる容易なプロバイダー切り替え
- **マルチプロバイダー対応 LLM ドキュメント分類・推論**:
  - **OpenRouter**: `qwen/qwen3.7-flash` 等のモデルを用いた高速・高精度なドキュメント構造化分類（`OpenRouterChatClassifier`）
  - **Mistral Chat**: `mistral-small-latest` 等の API による分類推論
  - **Ollama**: ローカル LLM による分類推論（`OllamaChatClassifier`）
  - `BaseChatClassifier` 抽象クラスおよび `create_chat_classifier` ファクトリ
- **安全な API キー管理**:
  - `python-dotenv` による `.env` ファイル（`MISTRAL_API_KEY`, `OPENROUTER_API_KEY`）からの自動環境変数ロード
  - `config.toml` に API キーを直書きせずに安全に利用可能なアーキテクチャ
- **ドキュメント解析 & 自動分類パイプライン**:
  - PDF（`.pdf`）に特化したテキスト抽出・AI分類・配置処理
  - PyMuPDF を利用した高速なデジタル PDF テキスト抽出
  - 書類の発行日・発行元・書類種別を考慮したインテリジェントなファイルリネームとフォルダ自動分類
- **メタデータログ出力**:
  - Polars を利用した分類メタデータの高速 CSV / TSV 出力
  - 実行日時タイムスタンプ付きログ保存機能
- **CLI & ユーザーインターフェース**:
  - Typer を採用した型安全で直感的なコマンドラインインターフェース
  - Rich との統合によるカラーログ出力、起動時設定パネル（OCR / Chat プロバイダー表示）、処理結果サマリーテーブル
  - Rich `Progress` によるリアルタイムな進捗バー（件数・パーセンテージ・経過時間）および現在フェーズのステータス表示
  - コマンドラインオプション: `-l, --limit`, `--provider`, `--chat-provider`, `--ocr/--no-ocr`, `--dry-run`, `--move`, `--copy`, `--recursive` 等

- **安全・柔軟な実行オプション**:
  - OCR スキップ（`--no-ocr` / `enable_ocr = false`）オプション: 画像スキャンに対するOCRをスキップし、埋め込みテキストレイヤーのみを抽出して爆速分類
  - 出力先同名ファイルとの衝突を防ぐ連番自動付与（`_1`, `_2`）ロジック
  - `try ... finally` 構造により、`Ctrl+C` や例外中断時でも処理済みレコードを確実に CSV / TSV ログに書き出し
