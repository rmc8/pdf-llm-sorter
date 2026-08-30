"""PDF LLM Sorter - メインエントリーポイント

OCRとLLM (Ollama) を活用してPDFファイルを解析・分類・整理するCLIアプリケーションのエントリーポイントです。
"""

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from pdf_llm_sorter.libs.config import AppConfig, load_config
from pdf_llm_sorter.libs.processor import DocumentProcessor, ProcessResult

console = Console()
logger = logging.getLogger("pdf_llm_sorter")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

app = typer.Typer(
    name="pdf-llm-sorter",
    help="PDF LLM Sorter - OCRとLLMを活用したPDF・画像ドキュメント分類・整理ツール",
    add_completion=False,
    rich_markup_mode="rich",
)


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


@app.command()
def run(
    inputs: Annotated[
        list[Path] | None,
        typer.Argument(
            help="処理対象のPDF/画像ファイルまたはディレクトリパス（未指定時は設定ファイルの input_folder を使用）",
        ),
    ] = None,
    config_path: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            help="設定ファイル (TOML) のパス",
        ),
    ] = DEFAULT_CONFIG_PATH,
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output",
            help="出力先フォルダパス（指定時は設定ファイルの output_folder を上書き）",
        ),
    ] = None,
    copy: Annotated[
        bool,
        typer.Option(
            "--copy",
            help="元ファイルを保持して出力先にコピーする (デフォルト)",
        ),
    ] = False,
    move: Annotated[
        bool,
        typer.Option(
            "--move",
            help="元ファイルを出力先へ移動（整理）する",
        ),
    ] = False,
    export_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help="結果メタデータの出力形式 (csv, tsv, both, none)",
        ),
    ] = None,
    no_timestamp: Annotated[
        bool,
        typer.Option(
            "--no-timestamp",
            help="出力ファイル名にタイムスタンプを付与せず固定ファイル名で保存する",
        ),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option(
            "-r",
            "--recursive",
            help="入力フォルダ配下のサブディレクトリを再帰的に走査する",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "-n",
            "--dry-run",
            help="実際のファイル移動/コピーを行わずに推論と配置先をシミュレーション表示する",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "-v",
            "--verbose",
            help="詳細なデバッグログを出力する",
        ),
    ] = False,
) -> None:
    """PDF LLM Sorter - OCRとLLMを活用してドキュメントを自動分類・整理します。"""
    setup_logging(verbose)

    # 設定ファイルの読み込み
    try:
        config: AppConfig = load_config(config_path)
        logger.debug("設定内容:\n%s", config.model_dump_json(indent=2))
    except Exception as e:
        logger.error("設定ファイルの読み込みに失敗しました: %s", e)
        raise typer.Exit(code=1) from e

    # CLI 引数による設定の上書き
    if output:
        config.file_system.output_folder = output
    if move:
        config.file_system.action_mode = "move"
    elif copy:
        config.file_system.action_mode = "copy"
    if export_format:
        config.file_system.export_format = export_format  # type: ignore[assignment]
    if no_timestamp:
        config.file_system.export_with_timestamp = False
    if recursive:
        config.file_system.recursive = True

    print_config_summary(config, dry_run=dry_run)

    try:
        processor = DocumentProcessor(config=config, dry_run=dry_run)
        results = processor.process_all(input_paths=inputs if inputs else None)

        print_results_table(results)

        success_count = sum(1 for r in results if r.status == "success")
        error_count = sum(1 for r in results if r.status == "error")

        if error_count == 0:
            console.print(
                f"[bold green]✔ 処理完了:[/bold green] 成功 {success_count} 件 / 合計 {len(results)} 件"
            )
        else:
            console.print(
                f"[bold red]✖ 処理完了:[/bold red] 成功 {success_count} 件 / [bold red]エラー {error_count} 件[/bold red] / 合計 {len(results)} 件"
            )
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        console.print(
            "[bold yellow]⚠ ユーザーによって処理が中断されました。[/bold yellow]"
        )
        raise typer.Exit(code=130) from None
    except typer.Exit:
        raise
    except Exception as e:
        logger.error("処理中に予期せぬエラーが発生しました: %s", e, exc_info=True)
        raise typer.Exit(code=1) from e


def main() -> None:
    """エントリーポイント関数。"""
    app()


if __name__ == "__main__":
    main()
