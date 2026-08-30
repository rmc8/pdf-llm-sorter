"""PDF LLM Sorter - メインエントリーポイント

OCRとLLM (Ollama) を活用してPDFファイルを解析・分類・整理するCLIアプリケーションのエントリーポイントです。
"""

import argparse
import logging
import sys
from pathlib import Path

from pdf_llm_sorter.libs.config import AppConfig, load_config
from pdf_llm_sorter.libs.processor import DocumentProcessor

logger = logging.getLogger("pdf_llm_sorter")


def setup_logging(verbose: bool = False) -> None:
    """ログ出力のフォーマットおよびログレベルを設定します。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを構築します。"""
    default_config_path = Path(__file__).resolve().parent / "config.toml"
    parser = argparse.ArgumentParser(
        prog="pdf_llm_sorter",
        description="PDF LLM Sorter - OCRとLLMを活用したPDF・画像ドキュメント分類・整理ツール",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config_path,
        help=f"設定ファイル (TOML) のパス (デフォルト: {default_config_path})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="出力先フォルダパス（指定時は設定ファイルの output_folder を上書き）",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="元ファイルを保持して出力先にコピーする (デフォルト)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="元ファイルを出力先へ移動（整理）する",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "both", "none"],
        default=None,
        help="結果メタデータの出力形式 (csv, tsv, both, none)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="出力ファイル名にタイムスタンプを付与せず固定ファイル名で保存する",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="入力フォルダ配下のサブディレクトリを再帰的に走査する",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="実際のファイル移動/コピーを行わずに推論と配置先をシミュレーション表示する",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="詳細なデバッグログを出力する",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="処理対象のPDF/画像ファイルまたはディレクトリパス（未指定時は設定ファイルの input_folder を使用）",
    )
    return parser


def main() -> int:
    """メイン実行関数。"""
    parser = create_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger.info("PDF LLM Sorter を起動しています...")

    # 設定ファイルの読み込み
    try:
        config: AppConfig = load_config(args.config)
        logger.info("設定ファイルを正常に読み込みました: %s", args.config)
        logger.debug("設定内容:\n%s", config.model_dump_json(indent=2))
    except Exception as e:
        logger.error("設定ファイルの読み込みに失敗しました: %s", e)
        return 1

    # CLI 引数による設定の上書き
    if args.output:
        config.file_system.output_folder = args.output
    if args.move:
        config.file_system.action_mode = "move"
    elif args.copy:
        config.file_system.action_mode = "copy"
    if args.format:
        config.file_system.export_format = args.format
    if args.no_timestamp:
        config.file_system.export_with_timestamp = False
    if args.recursive:
        config.file_system.recursive = True

    logger.info("Ollama Base URL : %s", config.ollama.base_url)
    logger.info("OCR Model       : %s", config.ollama.ocr_model or "(未設定)")
    logger.info("Chat Model      : %s", config.ollama.chat_model or "(未設定)")
    logger.info("Input Folder    : %s", config.file_system.input_folder)
    logger.info("Output Folder   : %s", config.file_system.output_folder)
    logger.info("Action Mode     : %s", config.file_system.action_mode)
    logger.info("Export Format   : %s", config.file_system.export_format)
    if args.dry_run:
        logger.info("Mode            : DRY RUN (シミュレーション)")

    try:
        processor = DocumentProcessor(config=config, dry_run=args.dry_run)
        results = processor.process_all(input_paths=args.inputs if args.inputs else None)

        success_count = sum(1 for r in results if r.status == "success")
        error_count = sum(1 for r in results if r.status == "error")

        logger.info(
            "=== 処理完了: 成功 %d 件 / エラー %d 件 / 合計 %d 件 ===",
            success_count,
            error_count,
            len(results),
        )

        return 0 if error_count == 0 else 1

    except KeyboardInterrupt:
        logger.warning("ユーザーによって処理が中断されました。")
        return 130
    except Exception as e:
        logger.error("処理中に予期せぬエラーが発生しました: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
