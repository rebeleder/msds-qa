from typing import Iterator

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatMessages(ChatMessageHistory):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[BaseMessage]

    def __getitem__(self, idx: int | slice) -> BaseMessage | list[BaseMessage]:
        return self.messages[idx]

    def __iter__(self) -> Iterator[BaseMessage]:
        return iter(self.messages)

    def get_messages(self) -> list[BaseMessage]:
        return self.messages

    def clear_messages(self) -> None:
        self.messages.clear()

    def get_ai_messages(self) -> list[str]:
        return [msg.content for msg in self.messages if isinstance(msg, AIMessage)]

    def get_human_messages(self) -> list[str]:
        return [msg.content for msg in self.messages if isinstance(msg, HumanMessage)]
