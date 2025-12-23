from langchain.agents import create_agent
from langchain_core.documents import Document 
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from puls_events_chatbot.services.fetch_data import clean_data, fetch_evenements_publics

import faiss
from mistralai import  Mistral
import os 
import pandas as pd
import faiss
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# ----------------- VARIABLES -------------------

api_key_mistral = os.getenv("MISTRAL_API_KEY")
api_key_hf = os.getenv("HUGGING_API_KEY")
os.environ["MISTRALAI_API_KEY"] = api_key_mistral
model = "mistral-medium-latest"
model_class = ChatMistralAI(model=model)
embedding_model ="mistral-embed"
client = Mistral(api_key=api_key_mistral)
#embedding_class = MistralAIEmbeddings(model=embedding_model)
embedding_class = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    huggingfacehub_api_token=api_key_hf
)
faiss_store = None
db_path = "./src/puls_events_chatbot/data/faiss_index"
df = None
backend_ready = "arret"

# ---------------- EMBEDDINGS ---------------

def get_embeddings_by_chunks(data):
    """
    data : liste de textes déjà nettoyés
    retour : liste d’embeddings (List[float]) dans le même ordre
    """
    global  df
    
    try:
        #chunk_size = 50
        #chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
        #embeddings_response = [client.embeddings.create(model=embedding_model, inputs=c) for c in chunks]
        #embeddings = [d.embedding for e in embeddings_response for d in e.data]
        
        embeddings = []
        embeddings.extend(embedding_class.embed_documents(data))
        print(len(data), len(df))
        assert len(embeddings) == len(df)
        df['embeddings'] = embeddings
    
    except Exception as e:
        print("Une erreur est survenue pendant l'embedding : "+str(e))

# --------------- BASE VECTORIELLE ----------------

def createdb():
    global faiss_store, df
    try:
        dimension = len(df["embeddings"].iloc[0]) 
        quantizer = faiss.IndexFlatL2(dimension)
        nlist = 1
        index_ivf  = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        index_ivf.train(np.array(df["embeddings"].tolist()).astype("float32"))
        index_ivf.add(np.array(df["embeddings"].tolist()).astype("float32"))
        index_ivf.nprobe=1

        df_no_emb = df.drop(columns=["embeddings"])
        documents = [
            Document(
                page_content=row["description"],
                metadata=row.to_dict()
            )
            for _, row in df_no_emb.iterrows()
        ]

        docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
        docstore = InMemoryDocstore(docstore_dict)

        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        faiss_store = FAISS(
            embedding_class,
            index_ivf,
            docstore,
            index_to_docstore_id
        )

        if len(df) != faiss_store.index.ntotal:
            raise ValueError("Données manquantes dans la bdd vectorielle.")
        
        print("Base de données vectorielle créée avec succès.")

    except Exception as e:
        print("Une erreur est survenue pendant création de la bdd : "+str(e))


# ------------------ INITIALISATION ---------------------------- 

def init():
    global df,backend_ready
    backend_ready = "en cours"
    df = fetch_evenements_publics()
    #df = clean_data()

    if df is not None : 
        get_embeddings_by_chunks(df["description"].tolist())
        print("Embedding Completed")
        if 'embeddings' in df.columns : 
            createdb()
    backend_ready = "actif"
    print(backend_ready)

def get_backend_status():
    return backend_ready

# --------------- FONCTION RETOURNANT LE CONTEXTE -----------------------

def metadata_to_str(current_question: str) -> str:
    docs = faiss_store.similarity_search(current_question, k=2)
    if not docs:
        return "Aucun événement correspondant."

    lines = []
    for d in docs:
        m = d.metadata
        
        # Extraction de chaque colonne
        url = m.get("URL", "")
        titre = m.get("Titre", "")
        description = m.get("description", "")
        description_longue = m.get("description_longue", "")
        image = m.get("image", "")
        thumbnail = m.get("thumbnail", "")
        date = m.get("date", "")
        premier_jour_debut = m.get("premier_jour_debut", "")
        premier_jour_fin = m.get("premier_jour_fin", "")
        dernier_jour_debut = m.get("dernier_jour_debut", "")
        dernier_jour_fin = m.get("dernier_jour_fin", "")
        nom_localisation = m.get("nom_localisation", "")
        adresse = m.get("adresse", "")
        code_postale = m.get("code_postale", "")
        ville = m.get("ville", "")
        telephone = m.get("telephone", "")
        site_web = m.get("site_web", "")
        description_localisation = m.get("description_localisation", "")
        lien_acces_en_ligne = m.get("lien_acces_en_ligne", "")
        age_minimum = m.get("age_minimum", "")
        age_maximum = m.get("age_maximum", "")
        source = m.get("source", "")

        # Construction du texte avec toutes les colonnes
        event_text = f"""Événement: {titre}
        Date: {date}
        Description: {description}
        Description longue: {description_longue}
        Lieu: {nom_localisation} - {adresse} {code_postale} {ville}
        Téléphone: {telephone}
        Site web: {site_web}
        Horaires: début {premier_jour_debut} fin {premier_jour_fin}
        Accès en ligne: {lien_acces_en_ligne}
        Âge: {age_minimum} à {age_maximum} ans
        Source: {source}
        URL: {url}
        Image: {image}
        Thumbnail: {thumbnail}"""

        lines.append(event_text.strip())

    return "\n\n".join(lines)
# --------------- CREATION DU PROMPT -----------------------

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    
    # Récupère la question 
    current_question = request.messages[-1].content

    # Récupère le contexte
    context = metadata_to_str(current_question)
    
    # PROMPT
    prompt = (
        "Tu es un assistant premium, un chatbot qui informe en donnant la sensation de converser avec un humain. "
        "Réponds à la question en utilisant uniquement les passages ci-dessous mais n’en fais pas mention. "
        "Si tu ne sais pas, excuse-toi et dis que tu ne sais pas. "
        "Tu peux saluer sans prendre en compte les passages.\n\n"
        f"Passages :\n{context}\n\n"
        f"Question :\n{current_question}\n\n"
        "Réponse :"
    )
    print("prompt enregistré !")
    print(prompt)
    return prompt

# --------------- CHATBOT LANGCHAIN -----------------------

def chat_with_mistral(query : str):
    # REQUETE POUR MISTRAL
    agent = create_agent(
        model=model_class,  
        tools=[],
        middleware=[prompt_with_context]
    )
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}) 
    print(f"Requete à mistral terminé ! {result}")
    return result["messages"][1].content