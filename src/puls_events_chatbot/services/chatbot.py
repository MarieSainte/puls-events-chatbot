from langchain.agents import create_agent
from langchain_core.documents import Document 
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_mistralai import ChatMistralAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from puls_events_chatbot.services.fetch_data import clean_data, fetch_evenements_publics

import faiss
import os 
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class PulsEventsRAG:
    """
    Classe regroupant l'ensemble de la logique métier du RAG : 
    - Configuration du LLM et de l'Embedding
    - Création et gestion de la base FAISS
    - Gestion du contexte et appel à l'Agent Langchain
    """
    def __init__(self):
        # ----------------- VARIABLES -------------------
        self.api_key_mistral = os.getenv("MISTRAL_API_KEY")
        self.api_key_hf = os.getenv("HUGGING_API_KEY")
        
        if self.api_key_mistral:
            os.environ["MISTRALAI_API_KEY"] = self.api_key_mistral
            
        self.model = "mistral-medium-latest"
        self.model_class = ChatMistralAI(model=self.model)
        
        self.embedding_class = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            huggingfacehub_api_token=self.api_key_hf
        )
        
        self.faiss_store = None
        self.df = None
        self.backend_ready = "arret"

    # ---------------- EMBEDDINGS ---------------
    def _get_embeddings_by_chunks(self, data):
        """Vectorise les descriptions et met à jour le DataFrame"""
        try:
            embeddings = []
            embeddings.extend(self.embedding_class.embed_documents(data))
            print(f"Nombre d'embeddings : {len(embeddings)}, Nombre de lignes : {len(self.df)}")
            self.df['embeddings'] = embeddings
        except Exception as e:
            print(f"Une erreur est survenue pendant l'embedding : {e}")

    # --------------- BASE VECTORIELLE ----------------
    def _createdb(self):
        """Crée l'index FAISS et le VectorStore depuis le DataFrame"""
        try:
            dimension = len(self.df["embeddings"].iloc[0]) 
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = 1
            index_ivf  = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            index_ivf.train(np.array(self.df["embeddings"].tolist()).astype("float32"))
            index_ivf.add(np.array(self.df["embeddings"].tolist()).astype("float32"))
            index_ivf.nprobe = 1

            df_no_emb = self.df.drop(columns=["embeddings"])
            documents = []
            for _, row in df_no_emb.iterrows():
                desc = row.get("description", "")
                page_content = str(desc) if pd.notna(desc) else ""
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=row.to_dict()
                    )
                )

            docstore_dict = {str(i): doc for i, doc in enumerate(documents)}
            docstore = InMemoryDocstore(docstore_dict)

            index_to_docstore_id = {i: str(i) for i in range(len(documents))}

            self.faiss_store = FAISS(
                self.embedding_class,
                index_ivf,
                docstore,
                index_to_docstore_id
            )

            if len(self.df) != self.faiss_store.index.ntotal:
                raise ValueError("Données manquantes dans la bdd vectorielle.")
            
            print("Base de données vectorielle créée avec succès.")

        except Exception as e:
            print(f"Une erreur est survenue pendant création de la bdd : {e}")

    # ------------------ INITIALISATION ---------------------------- 
    def init(self):
        """Point d'entrée principal pour démarrer le RAG"""
        self.backend_ready = "en cours"
        
        raw_df = fetch_evenements_publics()
        self.df = clean_data(raw_df)

        if self.df is not None and not self.df.empty: 
            data_to_embed = self.df["description"].astype(str).tolist()
            self._get_embeddings_by_chunks(data_to_embed)
            print("Embedding Completed")
            
            if 'embeddings' in self.df.columns: 
                self._createdb()
                
        self.backend_ready = "actif"
        print(f"self.backend_ready : {self.backend_ready}")

    def get_backend_status(self):
        return self.backend_ready

    # --------------- FONCTION RETOURNANT LE CONTEXTE -----------------------
    def metadata_to_str(self, current_question: str) -> str:
        """Cherche les événements correspondants dans FAISS"""
        if not self.faiss_store:
            return "Aucun événement correspondant."
            
        docs = self.faiss_store.similarity_search(current_question, k=2)
        if not docs:
            return "Aucun événement correspondant."

        lines = []
        for d in docs:
            m = d.metadata
            url = m.get("URL", "")
            titre = m.get("Titre", "")
            description = m.get("description", "")
            description_longue = m.get("description_longue", "")
            image = m.get("image", "")
            thumbnail = m.get("thumbnail", "")
            date = m.get("date", "")
            p_j_d = m.get("premier_jour_debut", "")
            p_j_f = m.get("premier_jour_fin", "")
            d_j_d = m.get("dernier_jour_debut", "")
            d_j_f = m.get("dernier_jour_fin", "")
            loc = m.get("nom_localisation", "")
            adresse = m.get("adresse", "")
            cp = m.get("code_postale", "")
            ville = m.get("ville", "")
            tel = m.get("telephone", "")
            site = m.get("site_web", "")
            lien = m.get("lien_acces_en_ligne", "")
            age_min = m.get("age_minimum", "")
            age_max = m.get("age_maximum", "")
            source = m.get("source", "")

            event_text = f"""Événement: {titre}
                Date: {date}
                Description: {description}
                Description longue: {description_longue}
                Lieu: {loc} - {adresse} {cp} {ville}
                Téléphone: {tel}
                Site web: {site}
                Horaires: début {p_j_d} fin {p_j_f}
                Accès en ligne: {lien}
                Âge: {age_min} à {age_max} ans
                Source: {source}
                URL: {url}
                Image: {image}
                Thumbnail: {thumbnail}"""
        
            lines.append(event_text.strip())

        return "\n\n".join(lines)

    # --------------- CHATBOT LANGCHAIN -----------------------
    def chat_with_mistral(self, query: str):
        try:
            @dynamic_prompt
            def prompt_with_context(request: ModelRequest) -> str:
                current_question = request.messages[-1].content
                context = self.metadata_to_str(current_question)
                
                prompt = (
                    "Tu es un assistant premium, un chatbot qui informe sur des evenements sur paris. "
                    "Réponds à la question en utilisant uniquement les passages ci-dessous mais n’en fais pas mention. "
                    "Si tu ne sais pas ou que les passages ne contiennent pas la réponse, excuse-toi et dis que tu ne sais pas et si la question est incohérente avec ton role, rappelle simplement ton role.\n\n"
                    "Voila un exemple de reponse pour question hors sujet :\n"
                    "Je suis un chatbot spécialisé dans l'évènementiel a paris, souhaite tu une information sur un évènement en particulier ?\n\n"
                    f"Passages :\n{context}\n\n"
                    f"Question :\n{current_question}\n\n"
                    "Réponse :"
                )
                print("prompt enregistré !")
                print(prompt.replace('\u2028', '\n'))
                return prompt

            agent = create_agent(
                model=self.model_class,  
                tools=[],
                middleware=[prompt_with_context]
            )
            result = agent.invoke({"messages": [{"role": "user", "content": query}]}) 
            print(f"Requete à mistral terminé ! {result}")
            return result["messages"][1].content
            
        except Exception as e:
            print(f"Erreur technique lors de l'appel à Mistral AI : {e}")
            return "Désolé, je rencontre actuellement un problème technique avec mon service d'intelligence artificielle. Veuillez réessayer plus tard."


#=================================
# Instance globale (Singleton)
#=================================
rag_system = PulsEventsRAG()

def init():
    return rag_system.init()

def get_backend_status():
    return rag_system.get_backend_status()

def metadata_to_str(query: str) -> str:
    return rag_system.metadata_to_str(query)

def chat_with_mistral(query: str) -> str:
    return rag_system.chat_with_mistral(query)