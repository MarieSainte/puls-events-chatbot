# POC Chatbot Intelligent - Puls-Events

Ce projet vise à démontrer la faisabilité technique et la pertinence métier d'un chatbot intelligent capable de répondre aux questions des utilisateurs sur des événements culturels à venir. Le système utilise une approche de **Retrieval-Augmented Generation** (RAG) combinant recherche vectorielle et génération de réponses en langage naturel.

## Objectifs

L'objectif principal est de créer un **Proof of Concept** (POC) fonctionnel pour Puls-Events, en développant un chatbot qui pourra répondre aux utilisateurs en s'appuyant sur une base de données d'événements culturels récents, accessibles via l'API **Open Agenda**. Le système doit intégrer un pipeline de recherche et de génération de réponses utilisant les technologies suivantes :

- **LangChain** : Pour la gestion du flux de données et l'interfaçage avec les modèles de langage.
- **Mistral** : Modèle de génération pour produire des réponses contextuelles et cohérentes.
- **Faiss** : Système de recherche vectorielle pour interroger les événements et récupérer les réponses pertinentes.

L'API REST exposera ce système et permettra aux utilisateurs d'envoyer des questions et de recevoir des réponses augmentées, générées à partir des données vectorisées.

## Structure du projet

Le projet est structuré de la manière suivante :

Interface frontend via gradio
Notebook : pour effectuer des tests rapidement
Services : 
    le chatbot (fait l'embedding et la communication avec mistral)
    fetch data (récupère les events sur l'api open agenda)
Models : le modele de la question de l'utilisateur
Data : les events au format CSV
Controllers : Endpoints pour communiquer avec l'api

## Instructions de reproduction

### Prérequis

**Docker**, **git** doivent être installés.

### Actions à réaliser

git clone https://github.com/MarieSainte/puls-events-chatbot.git
cd puls-events-chatbot

cp .env <-- ajoutez vos api_key mistral et huggingface comme ci-dessous
MISTRAL_API_KEY = "votre clé Mistral"
HUGGING_API_KEY = "votre clé Hugging Face"

docker build -t puls-chatbot .
docker run -d -p 8010:8010 --env-file .env --name chatbot puls-chatbot

L’API est accessible sur http://localhost:8010
Documentation interactive : http://localhost:8010/swagger

### Arrêt & nettoyage
docker stop chatbot
docker rm chatbot
docker image rm puls-chatbot

## Utilisation sans Docker

### Prérequis

**poetry**, **git** doivent être installés.

### Actions à réaliser 

git clone https://github.com/MarieSainte/puls-events-chatbot.git
cd puls-events-chatbot

cp .env <-- ajoutez vos clé api mistral et huggingface comme ci-dessous
MISTRAL_API_KEY = "xxxxxxx"
HUGGING_API_KEY = "hfxxxxxxx"

poetry install
poetry run uvicorn puls_events_chatbot:app --host 0.0.0.0 --port 5050 --reload --app-dir src

Lancer ensuite l’interface Gradio (démarre automatiquement sur http://127.0.0.1:7860).