# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier d'abord le fichier des dépendances pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du code dans le conteneur
COPY . .

# Informer Docker que l'app écoute sur le port 8010
EXPOSE 8010
EXPOSE 7860

# Configuration du PYTHONPATH pour que Python trouve le module src/puls_events_chatbot
ENV PYTHONPATH=/app/src

# Lancer le serveur avec Uvicorn
CMD ["uvicorn", "src.puls_events_chatbot.main:app", "--host", "0.0.0.0", "--port", "8010"]