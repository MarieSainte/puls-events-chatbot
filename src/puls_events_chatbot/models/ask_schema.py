from pydantic import BaseModel


class AskSchema(BaseModel):
    message: str