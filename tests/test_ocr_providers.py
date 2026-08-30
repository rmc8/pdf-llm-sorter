"""Mistral, OpenRouter, Ollama の OCR & Chat プロバイダーテスト"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf_llm_sorter.libs.chat import (
    MistralChatClassifier,
    OllamaChatClassifier,
    OpenRouterChatClassifier,
    create_chat_classifier,
)
from pdf_llm_sorter.libs.config import AppConfig, load_config
from pdf_llm_sorter.libs.ocr import MistralOCRClient, OllamaOCRClient, create_ocr_client


def test_create_ocr_client_factory():
    """create_ocr_client が設定されたプロバイダーに応じて適切なインスタンスを生成することをテスト"""
    config = AppConfig()

    config.ocr.provider = "mistral"
    client_mistral = create_ocr_client(config)
    assert isinstance(client_mistral, MistralOCRClient)

    config.ocr.provider = "ollama"
    client_ollama = create_ocr_client(config)
    assert isinstance(client_ollama, OllamaOCRClient)


def test_mistral_ocr_client_image_bytes():
    """MistralOCRClient が画像バイトから OCR テキストを抽出できることをテスト"""
    client = MistralOCRClient(api_key="test_mistral_key", max_retries=1)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "pages": [{"index": 0, "markdown": "## 請求書\n合計金額: 10,000円"}]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        result = client.extract_from_image_bytes(b"dummy_image_data")
        assert "請求書" in result
        assert "10,000円" in result
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_mistral_key"


def test_mistral_ocr_client_pdf_direct(tmp_path: Path):
    """MistralOCRClient が PDF ファイルを直接解析して DocumentOCRResult を返すことをテスト"""
    test_pdf = tmp_path / "dummy.pdf"
    test_pdf.write_bytes(b"%PDF-1.4 dummy pdf binary")

    client = MistralOCRClient(api_key="test_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "pages": [
            {"index": 0, "markdown": "1ページ目内容"},
            {"index": 1, "markdown": "2ページ目内容"},
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        doc_result = client.extract_from_pdf_file(test_pdf)
        assert doc_result is not None
        assert doc_result.total_pages == 2
        assert len(doc_result.pages) == 2
        assert "1ページ目内容" in doc_result.full_text
        assert "2ページ目内容" in doc_result.full_text


def test_mistral_ocr_client_retry():
    """MistralOCRClient のリトライ動作をテスト"""
    import httpx

    client = MistralOCRClient(api_key="test_key", max_retries=1)
    attempts = 0

    def mock_post(url, headers=None, json=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("Timeout")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"pages": [{"markdown": "成功テキスト"}]}
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        result = client.extract_from_image_bytes(b"image_bytes")
        assert attempts == 2
        assert result == "成功テキスト"


def test_load_config_with_new_sections():
    """config.toml が正しく読み込まれ、各プロバイダー設定が保持されることをテスト"""
    config = load_config()
    assert config.ocr.provider in ("mistral", "ollama")
    assert config.mistral.model == "mistral-ocr-latest"
    assert config.ollama.ocr_model == "deepseek-ocr:latest"
    assert config.openrouter.model == "qwen/qwen3.7-flash"


def test_create_chat_classifier():
    """create_chat_classifier が chat_provider に応じて適切なインスタンスを生成することをテスト"""
    config = AppConfig()

    config.general.chat_provider = "openrouter"
    clf_or = create_chat_classifier(config)
    assert isinstance(clf_or, OpenRouterChatClassifier)
    assert clf_or.provider_name == "openrouter"

    config.general.chat_provider = "mistral"
    clf_mis = create_chat_classifier(config)
    assert isinstance(clf_mis, MistralChatClassifier)
    assert clf_mis.provider_name == "mistral"

    config.general.chat_provider = "ollama"
    clf_ol = create_chat_classifier(config)
    assert isinstance(clf_ol, OllamaChatClassifier)


def test_openrouter_chat_classifier_classify():
    """OpenRouterChatClassifier が JSON レスポンスから FileModel を生成できることをテスト"""
    clf = OpenRouterChatClassifier(
        api_key="test_key",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen3.7-flash",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"file_name": "20260830_テスト社_請求書.pdf", "category": "領収書", "document_date": "2026-08-30", "issuer": "テスト社", "summary": "テスト要約", "tags": ["テスト"]}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res = clf.classify_document("テスト本文")
        assert res.file_name == "20260830_テスト社_請求書.pdf"
        assert res.category == "領収書"
        assert res.document_date == "2026-08-30"


def test_mistral_chat_classifier_classify():
    """MistralChatClassifier が JSON レスポンスから FileModel を生成できることをテスト"""
    clf = MistralChatClassifier(
        api_key="test_mistral_key",
        model="mistral-small-latest",
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"file_name": "20260830_ミストラル社_契約書.pdf", "category": "契約書", "document_date": "2026-08-30", "issuer": "ミストラル社", "summary": "契約要約", "tags": ["契約"]}'
                }
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        res = clf.classify_document("テスト契約書本文")
        assert res.file_name == "20260830_ミストラル社_契約書.pdf"
        assert res.category == "契約書"
        assert res.document_date == "2026-08-30"
        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_mistral_key"


def test_load_env_file_with_dotenv(tmp_path: Path):
    """load_env_file が python-dotenv を使用して .env から環境変数を正しく読み込むことをテスト"""
    import os

    from pdf_llm_sorter.libs.config import load_env_file

    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_KEY=sample_token_12345\n", encoding="utf-8")

    load_env_file(env_file)
    assert os.environ.get("TEST_DOTENV_KEY") == "sample_token_12345"
