from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv

from src.config import hp

dotenv.load_dotenv()
class SiliconflowClient:
    def __init__(self, base_url: str = hp.siliconflow_base_url, api_key: str = hp.siliconflow_api_key) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.chat_model = self.get_chat_model()
        self.embed_model = self.get_embed_model()

    def get_chat_model(
        self,
        chat_model_name: str = hp.siliconflow_chat_model,
    ) -> ChatOpenAI:
        chat_model = ChatOpenAI(model=chat_model_name, base_url=self.base_url)
        return chat_model

    def get_embed_model(
        self,
        embed_model_name: str = hp.siliconflow_embedding_model,
    ) -> OpenAIEmbeddings:
        embed_model = OpenAIEmbeddings(model=embed_model_name, base_url=self.base_url)
        return embed_model
