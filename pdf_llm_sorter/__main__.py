"""PDF LLM Sorter - メインエントリーポイント

OCRとLLM (Ollama) を活用してPDFファイルを解析・分類・整理するCLIアプリケーションのエントリーポイントです。
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from pdf_llm_sorter.libs.config import AppConfig, load_config
from pdf_llm_sorter.libs.processor import DocumentProcessor, ProcessResult

console = Console()
logger = logging.getLogger("pdf_llm_sorter")


def setup_logging(verbose: bool = False) -> None:
    """ログ出力のフォーマットおよびログレベルを設定します。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_path=verbose)
        ],
    )


def print_config_summary(config: AppConfig, dry_run: bool) -> None:
    """起動時の設定サマリーをリッチなテーブル/パネルで表示します。"""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Ollama Base URL", config.ollama.base_url)
    table.add_row("OCR Model", config.ollama.ocr_model or "(未設定)")
    table.add_row("Chat Model", config.ollama.chat_model or "(未設定)")
    table.add_row("Input Folder", config.file_system.input_folder)
    table.add_row("Output Folder", config.file_system.output_folder)
    table.add_row("Action Mode", config.file_system.action_mode)
    table.add_row("Export Format", config.file_system.export_format)
    table.add_row("Recursive Scan", "有効" if config.file_system.recursive else "無効")
    if dry_run:
        table.add_row("Mode", "[bold yellow]DRY RUN (シミュレーション)[/bold yellow]")

    console.print(
        Panel(
            table, title="[bold blue]PDF LLM Sorter 実行設定[/bold blue]", expand=False
        )
    )


def print_results_table(results: list[ProcessResult]) -> None:
    """処理結果一覧をリッチなテーブルで表示します。"""
    if not results:
        return

    table = Table(title="処理結果サマリー", show_lines=True)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("元ファイル", style="cyan")
    table.add_column("カテゴリ", style="magenta")
    table.add_column("決定ファイル名", style="green")
    table.add_column("状態", justify="center")

    for idx, r in enumerate(results, 1):
        if r.status == "success":
            status_str = "[bold green]成功[/bold green]"
        elif r.status == "skipped":
            status_str = "[yellow]スキップ[/yellow]"
        else:
            status_str = f"[bold red]エラー[/bold red]\n[dim]{r.error_message}[/dim]"

        table.add_row(
            str(idx),
            r.original_filename,
            r.category or "-",
            r.file_name or "-",
            status_str,
        )

    console.print(table)


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

    # 設定ファイルの読み込み
    try:
        config: AppConfig = load_config(args.config)
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

    print_config_summary(config, dry_run=args.dry_run)

    try:
        processor = DocumentProcessor(config=config, dry_run=args.dry_run)
        results = processor.process_all(
            input_paths=args.inputs if args.inputs else None
        )

        print_results_table(results)

        success_count = sum(1 for r in results if r.status == "success")
        error_count = sum(1 for r in results if r.status == "error")

        if error_count == 0:
            console.print(
                f"[bold green]✔ 処理完了:[/bold green] 成功 {success_count} 件 / 合計 {len(results)} 件"
            )
            return 0
        else:
            console.print(
                f"[bold red]✖ 処理完了:[/bold red] 成功 {success_count} 件 / [bold red]エラー {error_count} 件[/bold red] / 合計 {len(results)} 件"
            )
            return 1

    except KeyboardInterrupt:
        console.print(
            "[bold yellow]⚠ ユーザーによって処理が中断されました。[/bold yellow]"
        )
        return 130
    except Exception as e:
        logger.error("処理中に予期せぬエラーが発生しました: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
