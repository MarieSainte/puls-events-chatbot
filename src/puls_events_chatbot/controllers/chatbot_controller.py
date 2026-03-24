from fastapi import APIRouter, Body, Depends
from puls_events_chatbot.models.ask_schema import AskSchema
from puls_events_chatbot.services.chatbot import chat_with_mistral, get_backend_status, init


router = APIRouter(prefix = "/chatbot")

@router.post("/ask",response_model=None)
def chatbot_mistral(payload: AskSchema):
    """
    Envoi une requete augmentée au Chatbot
    - **query**: Texte contenant la question de l'utilisateur.
    si besoin : 
    Récupère les données via l'API OpenAgenda
    Relance l'initialisation de la base de données vectorielle
    """
    query = payload.message
    if get_backend_status() == "actif":
        print("statut du backend : pret")
        answer  = chat_with_mistral(query)
        return {"answer": answer, "code": ""}
    elif get_backend_status() == "arret" :
        print("statut du backend : en arret")
        init()
        return {"answer": "Le serveur démarre !", "code": ""}
    else : 
        print("statut du backend : en cours de démarrage")
        return {"answer": "Le serveur redémarre !", "code": ""}



@router.get("/rebuild",response_model=None)
def rebuild(username: str = Body(..., embed=True, examples=["admin"])):
    """
    Récupère les données via l'API OpenAgenda
    Relance l'initialisation de la base de données vectorielle
    """
    if username == "admin":
        init()
        return {"detail": "Base rechargée"}
    else:
        print("NON AUTORISE !")
        return {"detail": "Vous êtes pas autoriser à utiliser cette fonctionnalité."}


