from fastapi import APIRouter, Body, Depends
from puls_events_chatbot.models.ask_schema import AskSchema
from puls_events_chatbot.services.chatbot import chat_with_mistral, get_backend_status, init


router = APIRouter(prefix = "/chatbot")

@router.post("/ask",
    response_model=None,
    tags=["Chatbot"],
    summary="Interroger le chatbot RAG",
    description="""
        Envoi une requete augmentée au Chatbot
        - **Requete**: Texte contenant la question de l'utilisateur.
        si besoin : 
        Récupère les données via l'API OpenAgenda ou
        Relance l'initialisation de la base de données vectorielle
    """,
    responses={
        200: {
            "description": "Réponse du chatbot ou message d'état du système",
            "content": {
                "application/json": {
                    "examples": {
                        "requete_vide": {
                            "summary": "Requète vide",
                            "value": {
                                "answer": "Veuillez poser une question",
                                "code": ""
                            }
                        },
                        "reponse_normale": {
                            "summary": "Réponse RAG",
                            "value": {
                                "answer": "Bonjour ! Oui, il y a un événement qui pourrait vous intéresser : un Jobdating des Aides à domicile aura lieu...",
                                "code": ""
                            }
                        },
                        "serveur_demarrage": {
                            "summary": "Backend en cours d'initialisation",
                            "value": {
                                "answer": "Le serveur démarre !",
                                "code": ""
                            }
                        },
                        "serveur_restart": {
                            "summary": "Backend en cours de redémarrage",
                            "value": {
                                "answer": "Le serveur redémarre !",
                                "code": ""
                            }
                        }
                    }
                }
            }
        },
        422: {
            "description": "Erreur de validation du payload"
        }
    }
)
def chatbot_mistral(payload: AskSchema):
    """
    Endpoint principal du chatbot.
    """
    query = payload.message
    if query is None:
        return {"answer": "Veuillez poser une question", "code": ""}
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



@router.post("/rebuild",
    response_model=None,
    tags=["Chatbot"],
    summary="Recharger la base de données vectorielle",
    description="""
        Récupère les données via l'API OpenAgenda,
        Relance l'initialisation de la base de données vectorielle
    """
    )
def rebuild(username: str = Body(..., embed=True, example="admin", description="Nom d'utilisateur autorisé à relancer la base")):
    """
    Endpoint pour recharger la base de données vectorielle.
    """
    if username == "admin":
        init()
        return {"detail": "Base rechargée"}
    else:
        print("NON AUTORISE !")
        return {"detail": "Vous êtes pas autoriser à utiliser cette fonctionnalité."}


