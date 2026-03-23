import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Configuration des modèles
evaluator_llm = ChatMistralAI(
    model="mistral-large-latest", 
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

# Le modèle d'embedding
evaluator_embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    huggingfacehub_api_token=os.getenv("HUGGING_API_KEY")
)

SEUIL_PRECISION = 0.8

# Données de test 
df = pd.DataFrame({
    'question': ["Quels sont les événements à Paris ?"],
    'contexts': [["Le festival Jazz à Paris a lieu en juillet.", "L'exposition Louvre est ouverte."]],
    'ground_truth': ["Le festival Jazz et l'exposition au Louvre."],
    'answer': ["Il y a un festival de Jazz et une expo au Louvre."] 
})

dataset_test = Dataset.from_pandas(df)

# 2. Exécution de l'évaluation
result = evaluate(
    dataset_test,
    metrics=[context_precision],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

score_precision = result['context_precision']

# 3. Verdict pour la CI/CD
if score_precision < SEUIL_PRECISION:
    print(f"ÉCHEC : Score {score_precision:.4f} < {SEUIL_PRECISION}")
    sys.exit(1)
else:
    print(f"SUCCÈS : Score {score_precision:.4f} >= {SEUIL_PRECISION}")
    sys.exit(0)