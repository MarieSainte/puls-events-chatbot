import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from puls_events_chatbot.main import app
import gradio as gr

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

# --- TESTS FETCH DATA ---

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


def test_metadata_to_str_with_results():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG
    from langchain_core.documents import Document

    rag = PulsEventsRAG()
    rag.faiss_store = MagicMock()
    rag.faiss_store.similarity_search.return_value = [
        Document(
            page_content="Concert de jazz",
            metadata={
                "Titre": "Jazz Night",
                "description": "Concert de jazz à Paris",
                "date": "2026-04-10",
                "nom_localisation": "Salle X",
                "adresse": "10 rue test",
                "code_postale": "75001",
                "ville": "Paris",
                "URL": "http://example.com"
            }
        )
    ]

    result = rag.metadata_to_str("concert jazz")

    assert "Événement: Jazz Night" in result
    assert "Concert de jazz à Paris" in result
    assert "Paris" in result

def test_metadata_to_str_with_empty_results():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    rag.faiss_store = MagicMock()
    rag.faiss_store.similarity_search.return_value = []

    result = rag.metadata_to_str("concert")

    assert result == "Aucun événement correspondant."

@patch("puls_events_chatbot.services.chatbot.clean_data")
@patch("puls_events_chatbot.services.chatbot.fetch_evenements_publics")
def test_init_success(mock_fetch, mock_clean, mock_events_df):
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    mock_fetch.return_value = mock_events_df
    mock_clean.return_value = mock_events_df.copy()

    def fake_get_embeddings(data):
        rag.df["embeddings"] = [np.random.rand(384).tolist() for _ in range(len(rag.df))]

    rag._get_embeddings_by_chunks = MagicMock(side_effect=fake_get_embeddings)
    rag._createdb = MagicMock()

    rag.init()

    assert rag.backend_ready == "actif"
    rag._get_embeddings_by_chunks.assert_called_once()
    rag._createdb.assert_called_once()

def test_get_backend_status():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    assert rag.get_backend_status() == "arret"

def test_chat_with_mistral_success():
    from puls_events_chatbot.services.chatbot import PulsEventsRAG

    rag = PulsEventsRAG()
    rag.metadata_to_str = MagicMock(return_value="Contexte test")

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [
            {"role": "user", "content": "question"},
            MagicMock(content="Réponse générée")
        ]
    }

    with patch("puls_events_chatbot.services.chatbot.create_agent", return_value=mock_agent):
        result = rag.chat_with_mistral("Quels sont les événements ?")

    assert result == "Réponse générée"

@patch("puls_events_chatbot.services.chatbot.rag_system")
def test_module_get_backend_status(mock_rag_system):
    from puls_events_chatbot.services.chatbot import get_backend_status

    mock_rag_system.get_backend_status.return_value = "actif"
    result = get_backend_status()

    assert result == "actif"

@patch("puls_events_chatbot.services.chatbot.rag_system")
def test_module_metadata_to_str(mock_rag_system):
    from puls_events_chatbot.services.chatbot import metadata_to_str

    mock_rag_system.metadata_to_str.return_value = "Contexte"
    result = metadata_to_str("concert")

    assert result == "Contexte"

@patch("puls_events_chatbot.services.chatbot.rag_system")
def test_module_chat_with_mistral(mock_rag_system):
    from puls_events_chatbot.services.chatbot import chat_with_mistral

    mock_rag_system.chat_with_mistral.return_value = "Réponse"
    result = chat_with_mistral("concert")

    assert result == "Réponse"

@patch("puls_events_chatbot.services.chatbot.rag_system")
def test_module_init(mock_rag_system):
    from puls_events_chatbot.services.chatbot import init

    init()
    mock_rag_system.init.assert_called_once()

# --- TESTS DES ENDPOINTS (API) ---

def test_chatbot_ask_invalid_payload():
    response = client.post("/chatbot/ask", json={})
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

@patch("puls_events_chatbot.controllers.chatbot_controller.chat_with_mistral")
@patch("puls_events_chatbot.controllers.chatbot_controller.get_backend_status")
def test_chatbot_ask_backend_actif(mock_status, mock_chat):
    mock_status.return_value = "actif"
    mock_chat.return_value = "Voici un événement en Martinique"

    response = client.post(
        "/chatbot/ask",
        json={"message": "Quels sont les événements ?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Voici un événement en Martinique",
        "code": ""
    }

@patch("puls_events_chatbot.controllers.chatbot_controller.init")
@patch("puls_events_chatbot.controllers.chatbot_controller.get_backend_status")
def test_chatbot_ask_backend_arret(mock_status, mock_init):
    mock_status.return_value = "arret"

    response = client.post(
        "/chatbot/ask",
        json={"message": "Quels sont les événements ?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Le serveur démarre !",
        "code": ""
    }
    mock_init.assert_called_once()

@patch("puls_events_chatbot.controllers.chatbot_controller.get_backend_status")
def test_chatbot_ask_backend_en_cours(mock_status):
    mock_status.return_value = "en cours"

    response = client.post(
        "/chatbot/ask",
        json={"message": "Quels sont les événements ?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Le serveur redémarre !",
        "code": ""
    }

@patch("puls_events_chatbot.controllers.chatbot_controller.init")
def test_rebuild_admin(mock_init):
    response = client.request(
        "GET",
        "/chatbot/rebuild",
        json={"username": "admin"}
    )

    assert response.status_code == 200
    assert response.json() == {"detail": "Base rechargée"}
    mock_init.assert_called_once()

@patch("builtins.print")
def test_rebuild_unauthorized(mock_print):
    response = client.request(
        "GET",
        "/chatbot/rebuild",
        json={"username": "user"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "detail": "Vous êtes pas autoriser à utiliser cette fonctionnalité."
    }
    mock_print.assert_called_once_with("NON AUTORISE !")

def test_rebuild_missing_body():
    response = client.get("/chatbot/rebuild")
    assert response.status_code == 422