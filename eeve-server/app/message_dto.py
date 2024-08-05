from pydantic import BaseModel


class ChatUserInput(BaseModel):
    input: str
