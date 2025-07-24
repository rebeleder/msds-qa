from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import hp


class GeminiClient:
    def __init__(self, base_url: str = hp.gemini_base_url) -> None:
        self.base_url = base_url
        self.chat_model = self.get_chat_model()
        self.embed_model = self.get_embed_model()

    def get_chat_model(
        self, chat_model_name: str = hp.gemini_chat_model
    ) -> ChatGoogleGenerativeAI:
        chat_model = ChatGoogleGenerativeAI(model=chat_model_name)
        return chat_model

    def get_embed_model(
        self, embed_model_name: str = hp.gemini_embedding_model
    ) -> GoogleGenerativeAIEmbeddings:
        embed_model = GoogleGenerativeAIEmbeddings(model=embed_model_name)
        return embed_model
