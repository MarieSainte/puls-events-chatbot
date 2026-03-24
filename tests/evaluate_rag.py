import os
import sys
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

import json
from puls_events_chatbot.services.chatbot import PulsEventsRAG

# Configuration modèle
evaluator_llm = ChatMistralAI(
    model="mistral-large-latest", 
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

# modèle d'embedding
evaluator_embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    huggingfacehub_api_token=os.getenv("HUGGING_API_KEY")
)

SEUIL_PRECISION = 0.8

# Instanciation RAG
rag = PulsEventsRAG()

# Initialisation de la base de données vectorielle
if rag.get_backend_status() != "actif":
    print("Initialisation de la base vectorielle pour les tests...")
    rag.init()

# Chargement des données de test
json_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
with open(json_path, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

questions = []
contexts_list = []
ground_truths = []
answers = []

print("Génération des réponses pour l'évaluation...")
for case in test_cases:
    q = case["question"]
    gt = case["expected_response"]
    
    ans = rag.chat_with_mistral(q)
    ctx_str = rag.metadata_to_str(q)
    ctx_list = [ctx_str] if ctx_str and ctx_str != "Aucun événement correspondant." else []
    
    questions.append(q)
    contexts_list.append(ctx_list)
    ground_truths.append(gt)
    answers.append(ans)

df = pd.DataFrame({
    'question': questions,
    'contexts': contexts_list,
    'ground_truth': ground_truths,
    'answer': answers 
})

dataset_test = Dataset.from_pandas(df)

# Exécution de l'évaluation
metrics_to_run = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
]

result = evaluate(
    dataset_test,
    metrics=metrics_to_run,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

print("\n--- RÉSULTATS DES MÉTRIQUES (RAGAS) ---")
for metric_name, score in result.items():
    print(f"{metric_name}: {score:.4f}")

score_precision = result.get('context_precision', 0)

# Verdict pour la CI/CD
if score_precision < SEUIL_PRECISION:
    print(f"\nÉCHEC CI/CD : precision du contexte ({score_precision:.4f})  <  SEUIL ({SEUIL_PRECISION})")
    sys.exit(1)
else:
    print(f"\nSUCCÈS CI/CD : precision du contexte ({score_precision:.4f})  >=  SEUIL ({SEUIL_PRECISION})")
    sys.exit(0)