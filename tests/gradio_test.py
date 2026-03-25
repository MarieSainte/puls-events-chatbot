import pytest
from unittest.mock import MagicMock, patch
import gradio as gr
import os

os.environ["API_BASE"] = "http://test-api/chatbot"

# --- GRADIO TESTS ---

@patch("puls_events_chatbot.gradio_interface.requests.post")
def test_gradio_chat_success_with_code(mock_post):
    from puls_events_chatbot.gradio_interface import chat

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "answer": "Voici la réponse",
        "code": "print('hello')"
    }
    mock_post.return_value = mock_response

    result = chat("Montre-moi du code")

    assert "Voici la réponse" in result
    assert "```python" in result
    assert "print('hello')" in result

@patch("puls_events_chatbot.gradio_interface.requests.post")
def test_gradio_chat_api_error(mock_post):
    from puls_events_chatbot.gradio_interface import chat

    mock_post.side_effect = Exception("API down")

    result = chat("Quels sont les événements ?")

    assert "Erreur lors de l'appel à l'API" in result
    assert "API down" in result

@patch("puls_events_chatbot.gradio_interface.requests.post")
def test_gradio_rebuild_api_success(mock_post):
    from puls_events_chatbot.gradio_interface import rebuild_api

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    rebuild_api("admin")

    mock_post.assert_called_once()

@patch("puls_events_chatbot.gradio_interface.requests.post")
def test_gradio_rebuild_api_error(mock_post):
    from puls_events_chatbot.gradio_interface import rebuild_api

    mock_post.side_effect = Exception("Rebuild failed")

    # This should not raise but print the error
    rebuild_api("admin")

