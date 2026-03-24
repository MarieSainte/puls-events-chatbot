from pydantic import BaseModel, Field


class AskSchema(BaseModel):
    message: str = Field(
        ..., 
        description="Question de l'utilisateur pour le chatbot", 
        examples=["je recherche un emploi en alternance, y a t il un evenement prochainement ?"]
    )