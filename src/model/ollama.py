from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.config import hp


class OllamaClient:
    def __init__(self, base_url: str = hp.ollama_host) -> None:
        self.base_url = base_url
        self.chat_model, self.embed_model = self.get_chat_embed_model()

    def get_chat_embed_model(
        self,
        chat_model_name: str = hp.ollama_chat_model,
        embed_model_name: str = hp.ollama_embedding_model,
    ) -> tuple[ChatOllama, OllamaEmbeddings]:
        chat_model = ChatOllama(
            model=chat_model_name,
            base_url=self.base_url,
        )
        embed_model = OllamaEmbeddings(
            model=embed_model_name,
            base_url=self.base_url,
        )
        return chat_model, embed_model
