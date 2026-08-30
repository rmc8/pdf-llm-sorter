"""エラーハンドリング・文字数上限・中断時ログ出力のテスト"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from pdf_llm_sorter.libs.chat import truncate_document_text
from pdf_llm_sorter.libs.config import AppConfig
from pdf_llm_sorter.libs.ocr import extract_text_from_pdf
from pdf_llm_sorter.libs.processor import DocumentProcessor, ProcessResult


def test_truncate_document_text_short():
    """短いテキストは切り詰められずにそのまま返されることをテスト"""
    short_text = "これは短いドキュメントテキストです。"
    result = truncate_document_text(short_text, max_chars=100)
    assert result == short_text


def test_truncate_document_text_long():
    """長文テキストが先頭と末尾を維持しつつ指定文字数以下に切り詰められることをテスト"""
    long_text = "START_" + "A" * 5000 + "_MIDDLE_" + "B" * 5000 + "_END"
    max_chars = 1000
    result = truncate_document_text(long_text, max_chars=max_chars)

    assert len(result) <= max_chars
    assert result.startswith("START_")
    assert result.endswith("_END")
    assert "... [中略: ドキュメント中間部分を省略] ..." in result


def test_extract_text_from_pdf_max_pages(tmp_path: Path):
    """PDFページ数制限 (max_pages) が正しく適用されることをテスト"""
    # 3ページのテスト用PDFを作成
    pdf_file = tmp_path / "multi_page_test.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text(
            (50, 50), f"Page {i + 1} content text here with enough length."
        )
    doc.save(str(pdf_file))
    doc.close()

    # max_pages=1 で抽出
    res1 = extract_text_from_pdf(pdf_file, max_pages=1)
    assert res1.total_pages == 3
    assert len(res1.pages) == 1
    assert "Page 1" in res1.full_text
    assert "Page 2" not in res1.full_text

    # max_pages=2 で抽出
    res2 = extract_text_from_pdf(pdf_file, max_pages=2)
    assert len(res2.pages) == 2
    assert "Page 1" in res2.full_text
    assert "Page 2" in res2.full_text
    assert "Page 3" not in res2.full_text

    # max_pages=0 または None で全ページ抽出
    res_all = extract_text_from_pdf(pdf_file, max_pages=0)
    assert len(res_all.pages) == 3


def test_process_all_finally_exports_on_exception(tmp_path: Path):
    """処理途中で例外（または中断）が発生した場合でも finally 節で export_results が呼ばれることをテスト"""
    config = AppConfig()
    config.file_system.output_folder = str(tmp_path / "output")
    config.file_system.export_path = str(tmp_path / "output" / "results.csv")

    file1 = tmp_path / "doc1.pdf"
    file2 = tmp_path / "doc2.pdf"
    file1.touch()
    file2.touch()

    processor = DocumentProcessor(config=config, dry_run=False)

    call_count = 0

    def mock_process_file(file_path: Path, **kwargs) -> ProcessResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt("ユーザーによる中断シミュレーション")
        return ProcessResult(
            original_path=str(file_path),
            original_filename=file_path.name,
            category="領収書",
            file_name=f"renamed_{file_path.name}",
            status="success",
        )

    processor.process_file = mock_process_file  # type: ignore[assignment]
    processor.scan_inputs = lambda paths=None: [file1, file2]  # type: ignore[assignment]

    with patch.object(processor, "export_results") as mock_export:
        with pytest.raises(KeyboardInterrupt):
            processor.process_all([file1, file2])

        # 1件目の処理済みレコードが export_results に渡されて呼び出されていることを検証
        mock_export.assert_called_once()
        saved_results = mock_export.call_args[0][0]
        assert len(saved_results) == 1
        assert saved_results[0].original_filename == "doc1.pdf"


def test_chat_classifier_retry_on_failure():
    """ChatClassifier が 1回目失敗時にリトライして 2回目で成功することをテスト"""
    from langchain_core.messages import AIMessage

    from pdf_llm_sorter.libs.chat import OllamaChatClassifier

    classifier = OllamaChatClassifier(max_retries=1, timeout=5.0)

    attempts = 0

    def mock_invoke(messages):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("1回目の通信タイムアウト")
        return AIMessage(
            content='{"file_name": "20260401_領収書.pdf", "category": "領収書", "document_date": "2026-04-01", "issuer": "テスト社", "summary": "要約", "tags": ["タグ"]}'
        )

    with patch("langchain_ollama.ChatOllama.invoke", side_effect=mock_invoke):
        result = classifier.classify_document(
            document_text="テスト本文", original_filename="test.pdf"
        )
        assert attempts == 2
        assert result.file_name == "20260401_領収書.pdf"
        assert result.category == "領収書"


def test_ocr_client_retry_on_failure():
    """OllamaOCRClient が 1回目失敗時にリトライして 2回目で成功することをテスト"""
    import httpx

    from pdf_llm_sorter.libs.ocr import OllamaOCRClient

    client = OllamaOCRClient(max_retries=1, timeout=5.0)

    attempts = 0

    def mock_post(url, json=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("OCRタイムアウト")
        # 成功レスポンスのモック
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "抽出されたOCRテキスト"}
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        text = client.extract_from_image_bytes(b"dummy_bytes")
        assert attempts == 2
        assert text == "抽出されたOCRテキスト"
