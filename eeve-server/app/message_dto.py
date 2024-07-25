from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
class MessageBase(BaseModel):
    content: str


class HumanMessage(MessageBase):
    type: str = "human"


class AIMessage(MessageBase):
    type: str = "ai"
    tool_calls: List = []
    invalid_tool_calls: List = []
    usage_metadata: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }


class SystemMessage(MessageBase):
    type: str = "system"


class Input_ms(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]