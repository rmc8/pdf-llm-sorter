"""PDF LLM Sorter - メインエントリーポイント

OCRとLLM (Ollama) を活用してPDFファイルを解析・分類・整理するCLIアプリケーションのエントリーポイントです。
"""

import argparse
import logging
import sys
from pathlib import Path

from pdf_llm_sorter.libs.config import AppConfig, load_config

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
        description="PDF LLM Sorter - OCRとLLMを活用したPDF分類・整理ツール",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=default_config_path,
        help=f"設定ファイル (TOML) のパス (デフォルト: {default_config_path})",
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
        help="処理対象のPDFファイルまたはディレクトリパス",
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
        logger.info("Ollama Base URL : %s", config.ollama.base_url)
        logger.info("OCR Model       : %s", config.ollama.ocr_model or "(未設定)")
        logger.info("Chat Model      : %s", config.ollama.chat_model or "(未設定)")
    except Exception as e:
        logger.error("設定ファイルの読み込みに失敗しました: %s", e)
        return 1

    # 入力ファイルの確認（将来の処理のプレースホルダー）
    if args.inputs:
        logger.info("処理対象の入力 (%d 件):", len(args.inputs))
        for input_path in args.inputs:
            logger.info(
                " - %s (存在: %s)",
                input_path,
                "あり" if input_path.exists() else "なし",
            )
    else:
        logger.info("処理対象のPDFファイル/ディレクトリは指定されていません。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
