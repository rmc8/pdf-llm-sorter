"""PDFおよび画像ファイルを走査・解析・分類・配置し、Polarsで結果を出力するパイプラインモジュール。"""

from datetime import datetime
import logging
from pathlib import Path
import shutil
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from pdf_llm_sorter.libs.chat import OllamaChatClassifier
from pdf_llm_sorter.libs.config import AppConfig
from pdf_llm_sorter.libs.model import FileModel
from pdf_llm_sorter.libs.ocr import OllamaOCRClient, extract_text_from_pdf

logger = logging.getLogger("pdf_llm_sorter.processor")

SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS


class ProcessResult(BaseModel):
    """単一ファイルの処理・分類結果メタデータ"""

    original_path: str = Field(description="元ファイルのフルパス")
    original_filename: str = Field(description="元ファイル名")
    output_path: str = Field(default="", description="配置先ファイルのフルパス")
    category: str = Field(default="", description="分類カテゴリ名")
    file_name: str = Field(default="", description="決定されたファイル名")
    document_date: str = Field(default="", description="書類の日付 (YYYY-MM-DD)")
    issuer: str = Field(default="", description="発行元・組織名")
    summary: str = Field(default="", description="書類の要約")
    tags: str = Field(default="", description="カンマ区切りのタグ")
    status: Literal["success", "skipped", "error"] = Field(
        default="success", description="処理ステータス"
    )
    error_message: str = Field(default="", description="エラー発生時のメッセージ")
    processed_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="処理実行日時",
    )


def get_unique_destination_path(destination: Path) -> Path:
    """同名ファイルが存在する場合に連番（_1, _2 ...）を付与したユニークな出力先パスを返します。"""
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class DocumentProcessor:
    """PDF・画像ドキュメントの分類・配置・メタデータ出力パイプライン"""

    def __init__(
        self,
        config: AppConfig,
        ocr_client: OllamaOCRClient | None = None,
        chat_classifier: OllamaChatClassifier | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.dry_run = dry_run

        self.ocr_client = ocr_client or OllamaOCRClient.from_config(config.ollama)
        self.chat_classifier = chat_classifier or OllamaChatClassifier.from_config(config)

    def scan_inputs(self, input_paths: list[Path] | None = None) -> list[Path]:
        """指定されたパス一覧または設定の input_folder から処理対象のファイル一覧を収集します。"""
        targets: list[Path] = []

        if not input_paths:
            # 設定ファイルの input_folder を参照
            folder_str = self.config.file_system.input_folder
            input_dir = Path(folder_str).expanduser().resolve()
            if not input_dir.exists():
                logger.warning("input_folder が存在しません: %s", input_dir)
                return []
            input_paths = [input_dir]

        for path in input_paths:
            path = path.expanduser().resolve()
            if not path.exists():
                logger.warning("指定されたパスが存在しません: %s", path)
                continue

            if path.is_file():
                if path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
                    targets.append(path)
                else:
                    logger.debug(
                        "対象外の拡張子のファイルをスキップします: %s", path.name
                    )
            elif path.is_dir():
                pattern = "**/*" if self.config.file_system.recursive else "*"
                for child in path.glob(pattern):
                    if child.is_file() and child.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
                        targets.append(child)

        # 重複除去とソート
        unique_targets = sorted(list(dict.fromkeys(targets)))
        logger.info("対象ファイルを %d 件検出しました。", len(unique_targets))
        return unique_targets

    def extract_text(self, file_path: Path) -> str:
        """ファイル形式に応じたテキスト抽出を実行します。"""
        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_PDF_EXTENSIONS:
            result = extract_text_from_pdf(
                pdf_path=file_path,
                ocr_client=self.ocr_client,
            )
            return result.full_text
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
            logger.info("画像ファイル OCR 実行中: %s", file_path.name)
            return self.ocr_client.extract_from_image_file(file_path)
        else:
            raise ValueError(f"未対応のファイル拡張子です: {suffix}")

    def process_file(self, file_path: Path) -> ProcessResult:
        """1つのファイルを処理（OCR・LLM分類・フォルダ作成・配置）します。"""
        logger.info("=== 処理開始: %s ===", file_path.name)
        orig_path_str = str(file_path)
        orig_name = file_path.name

        try:
            # 1. テキスト抽出 (PDF テキストレイヤー / Vision OCR)
            extracted_text = self.extract_text(file_path)
            if not extracted_text.strip():
                logger.warning("テキストが抽出できませんでした: %s", orig_name)

            # 2. LLM 分類・リネーム名決定
            classification: FileModel = self.chat_classifier.classify_document(
                document_text=extracted_text,
                original_filename=orig_name,
            )

            # 3. 出力先ディレクトリ・パスの準備（フォルダがない場合は自動作成）
            output_base = (
                Path(self.config.file_system.output_folder).expanduser().resolve()
            )
            category_dir = output_base / classification.category
            if not self.dry_run:
                category_dir.mkdir(parents=True, exist_ok=True)

            target_filename = classification.file_name
            # 元ファイルが画像の場合は、拡張子を保持またはPDFリネームを調整
            if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                if not target_filename.lower().endswith(file_path.suffix.lower()):
                    target_filename = f"{Path(target_filename).stem}{file_path.suffix.lower()}"

            initial_dest = category_dir / target_filename
            final_dest = get_unique_destination_path(initial_dest)

            # 4. ファイル配置 (copy / move)
            action_mode = self.config.file_system.action_mode
            if self.dry_run:
                logger.info(
                    "[Dry Run] 配置シミュレーション: %s -> %s (action=%s)",
                    file_path,
                    final_dest,
                    action_mode,
                )
            else:
                if action_mode == "move":
                    logger.info("ファイル移動: %s -> %s", file_path, final_dest)
                    shutil.move(file_path, final_dest)
                else:
                    logger.info("ファイルコピー: %s -> %s", file_path, final_dest)
                    shutil.copy2(file_path, final_dest)

            tags_str = ", ".join(classification.tags) if classification.tags else ""

            return ProcessResult(
                original_path=orig_path_str,
                original_filename=orig_name,
                output_path=str(final_dest) if not self.dry_run else str(initial_dest),
                category=classification.category,
                file_name=final_dest.name if not self.dry_run else initial_dest.name,
                document_date=classification.document_date,
                issuer=classification.issuer,
                summary=classification.summary,
                tags=tags_str,
                status="success",
            )

        except Exception as e:
            logger.error("処理中にエラーが発生しました (%s): %s", orig_name, e, exc_info=True)
            return ProcessResult(
                original_path=orig_path_str,
                original_filename=orig_name,
                status="error",
                error_message=str(e),
            )

    def process_all(self, input_paths: list[Path] | None = None) -> list[ProcessResult]:
        """全対象ファイルを順次処理し、結果をまとめてCSV/TSVに出力します。"""
        targets = self.scan_inputs(input_paths)
        if not targets:
            logger.info("処理対象のファイルがありませんでした。")
            return []

        results: list[ProcessResult] = []
        for idx, file_path in enumerate(targets, 1):
            logger.info("[%d/%d] ファイル処理中: %s", idx, len(targets), file_path.name)
            res = self.process_file(file_path)
            results.append(res)

        # 結果を Polars でエクスポート
        if results and not self.dry_run:
            self.export_results(results)

        return results

    def export_results(self, results: list[ProcessResult]) -> list[Path]:
        """Polars を使用して処理結果を CSV / TSV 形式で出力します。"""
        export_fmt = self.config.file_system.export_format
        if export_fmt == "none":
            logger.info("export_format が 'none' に設定されているためエクスポートをスキップします。")
            return []

        # 新規結果レコードの辞書リスト作成
        new_data: list[dict[str, Any]] = [r.model_dump() for r in results]
        new_df = pl.DataFrame(new_data)

        # タイムスタンプ文字列の生成
        now = datetime.now()
        ts_str = now.strftime("%Y%m%d_%H%M%S")
        date_str = now.strftime("%Y%m%d")

        output_base = Path(self.config.file_system.output_folder).expanduser().resolve()
        output_base.mkdir(parents=True, exist_ok=True)

        formats_to_export: list[str] = []
        if export_fmt in ("csv", "both"):
            formats_to_export.append("csv")
        if export_fmt in ("tsv", "both"):
            formats_to_export.append("tsv")

        exported_files: list[Path] = []
        raw_export_path = self.config.file_system.export_path.strip()
        with_timestamp = self.config.file_system.export_with_timestamp

        for fmt in formats_to_export:
            sep = "," if fmt == "csv" else "\t"

            if raw_export_path:
                # プレースホルダーの置換 ({timestamp}, {datetime}, {date})
                formatted_path = (
                    raw_export_path.replace("{timestamp}", ts_str)
                    .replace("{datetime}", ts_str)
                    .replace("{date}", date_str)
                )
                target_file = Path(formatted_path).expanduser()
                is_directory = (
                    target_file.is_dir()
                    or raw_export_path.endswith(("/", "\\"))
                    or target_file.suffix == ""
                )

                if is_directory:
                    base_name = f"classification_results_{ts_str}.{fmt}" if with_timestamp else f"classification_results.{fmt}"
                    target_file = target_file / base_name
                else:
                    if target_file.suffix.lower() != f".{fmt}":
                        target_file = target_file.with_suffix(f".{fmt}")
                    # プレースホルダーが含まれておらず、かつタイムスタンプ付与が有効な場合はファイル名に付与
                    if with_timestamp and not any(p in raw_export_path for p in ["{timestamp}", "{datetime}", "{date}"]):
                        stem = target_file.stem
                        target_file = target_file.with_name(f"{stem}_{ts_str}.{fmt}")
            else:
                base_name = f"classification_results_{ts_str}.{fmt}" if with_timestamp else f"classification_results.{fmt}"
                target_file = output_base / base_name

            target_file.parent.mkdir(parents=True, exist_ok=True)

            # 既存ファイルがある場合は結合 (マージ)
            if target_file.exists():
                try:
                    existing_df = pl.read_csv(target_file, separator=sep)
                    # 既存の DataFrame と結合
                    combined_df = pl.concat([existing_df, new_df], how="diagonal")
                    # original_path に基づいて重複排除（最新を保持）
                    if "original_path" in combined_df.columns:
                        combined_df = combined_df.unique(
                            subset=["original_path"], keep="last"
                        )
                except Exception as e:
                    logger.warning(
                        "既存の %s の読み込みに失敗したため新規上書きします: %s",
                        target_file,
                        e,
                    )
                    combined_df = new_df
            else:
                combined_df = new_df

            combined_df.write_csv(target_file, separator=sep)
            logger.info(
                "結果を %s にエクスポートしました (計 %d 行)", target_file, len(combined_df)
            )
            exported_files.append(target_file)

        return exported_files
