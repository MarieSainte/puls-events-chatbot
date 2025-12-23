FROM python:3.10-slim AS builder

# Définir un répertoire de travail
WORKDIR /app

# Installer Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copier config Poetry
COPY pyproject.toml poetry.lock* ./

# Installer dépendances sans installer le projet en mode package
RUN poetry install --no-root

# Copier le code
COPY . .

# Exposer le port 
EXPOSE 8010:8010

# Lancer le serveur
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]