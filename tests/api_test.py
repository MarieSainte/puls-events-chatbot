import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from puls_events_chatbot.main import app

client = TestClient(app)



@pytest.fixture
def mock_events_df():
    return pd.DataFrame({
        "Titre": ["Event 1"],
        "description": ["Description test"],
        "URL": ["http://test.com"],
        "date": ["2025-01-01"],
        "nom_localisation": ["Fort-de-France"],
        "adresse": ["Rue test"],
        "code_postale": ["97200"],
        "ville": ["FDF"]
    })

# --- TESTS DES SERVICES ---

def test_clean_data(mock_events_df):
    from puls_events_chatbot.services.fetch_data import clean_data
    df_with_doubles = pd.concat([mock_events_df, mock_events_df])
    cleaned_df = clean_data(df_with_doubles)
    assert len(cleaned_df) == 1

def test_clean_data_empty_dataframe():
    from puls_events_chatbot.services.fetch_data import clean_data

    empty_df = pd.DataFrame()
    cleaned_df = clean_data(empty_df)

    assert cleaned_df.empty

@patch("puls_events_chatbot.services.fetch_data.requests.get")
def test_fetch_evenements_publics_success(mock_get):
    from puls_events_chatbot.services.fetch_data import fetch_evenements_publics

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [
            {
                "Titre": "Test",
                "description": "Description test"
            }
        ]
    }
    mock_get.return_value = mock_response

    df = fetch_evenements_publics()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Titre" in df.columns

# --- TESTS DU CHATBOT ---

def test_get_embeddings_by_chunks(mock_events_df):
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    rag.df = mock_events_df.copy()
    rag.embedding_class = MagicMock()
    rag.embedding_class.embed_documents.return_value = [np.random.rand(384).tolist()]

    rag._get_embeddings_by_chunks(rag.df["description"].tolist())

    assert "embeddings" in rag.df.columns

def test_createdb(mock_events_df):
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    rag.df = mock_events_df.copy()
    rag.df["embeddings"] = [np.random.rand(384).tolist()]

    rag._createdb()

    assert rag.faiss_store is not None

def test_metadata_to_str_without_store():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    result = rag.metadata_to_str("concert")
    assert result == "Aucun événement correspondant."


def test_chat_with_mistral_returns_fallback_on_error():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    rag.model_class = MagicMock()

    with patch("puls_events_chatbot.services.chatbot.create_agent") as mock_create_agent:
        mock_create_agent.side_effect = Exception("Erreur Mistral")

        result = rag.chat_with_mistral("Quels sont les événements ?")

    assert result == (
        "Désolé, je rencontre actuellement un problème technique avec mon service "
        "d'intelligence artificielle. Veuillez réessayer plus tard."
    )
# --- TESTS DES ENDPOINTS (API) ---

def test_chatbot_ask_invalid_payload():
    response = client.post("/chatbot/ask", json={})
    assert response.status_code == 422

def test_chatbot_ask_endpoint_starting_state():
    response = client.post(
        "/chatbot/ask",
        json={"message": "Quels sont les événements ?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()

def test_rebuild_invalid_request():
    response = client.get("/chatbot/rebuild", params={"username": "user"})
    assert response.status_code == 422

def test_rebuild_missing_params():
    response = client.get("/chatbot/rebuild")
    assert response.status_code == 422

def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "The application is running."
    }