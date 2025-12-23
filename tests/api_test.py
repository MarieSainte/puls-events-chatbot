import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from puls_events_chatbot.__init__ import app  

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
    # On ajoute des doublons pour tester le nettoyage
    df_with_doubles = pd.concat([mock_events_df, mock_events_df])
    cleaned_df = clean_data(df_with_doubles)
    assert len(cleaned_df) == 1

@patch("puls_events_chatbot.services.chatbot.requests.get")
def test_fetch_evenements_publics(mock_get):
    from puls_events_chatbot.services.fetch_data import fetch_evenements_publics
    # Simulation d'une réponse API réussie
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"results": [{"Titre": "Test"}]}
    
    df = fetch_evenements_publics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# --- TESTS DU CHATBOT (LOGIQUE RAG) ---

@patch("puls_events_chatbot.services.chatbot.embedding_class.embed_documents")
def test_vector_store_initialization(mock_embed, mock_events_df):
    from puls_events_chatbot.services.chatbot import createdb, get_embeddings_by_chunks
    import puls_events_chatbot.services.chatbot as chatbot
    
    # Mock des embeddings (vecteur de taille 384 par ex)
    chatbot.df = mock_events_df
    mock_embed.return_value = [np.random.rand(384).tolist()]
    
    get_embeddings_by_chunks(mock_events_df["description"].tolist())
    created_db = createdb()
    
    assert chatbot.faiss_store is not None
    assert "embeddings" in chatbot.df.columns

# --- TESTS DES ENDPOINTS (API) ---

def test_get_status():
    response = client.get("/chatbot/ask") # Simulation d'un appel pour voir l'état
    # Si le backend est à l'arrêt, il devrait initier le démarrage
    assert response.status_code in [200, 405]

@patch("puls_events_chatbot.services.chatbot.chat_with_mistral")
def test_chatbot_ask_endpoint(mock_chat):
    # On simule une réponse de Mistral
    mock_chat.return_value = "Voici un événement en Martinique"
    
    # On force le statut à actif pour le test
    with patch("puls_events_chatbot.services.chatbot.get_backend_status", return_value="actif"):
        response = client.post(
            "/chatbot/ask",
            json={"message": "Quels sont les événements ?"}
        )
        
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] == "Voici un événement en Martinique"

def test_rebuild_unauthorized():
    response = client.get("/chatbot/rebuild", json={"username": "user"})
    assert response.json()["detail"] == "Vous êtes pas autoriser à utiliser cette fonctionnalité."