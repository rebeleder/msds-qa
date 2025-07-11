from dataclasses import dataclass


@dataclass
class hp:
    ollama_host: str = "http://192.168.215.3:11434"
    ollama_chat_model: str = "qwen3:0.6b"
    ollama_embedding_model: str = "nomic-embed-text:latest"

    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    # siliconflow_chat_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    siliconflow_chat_model: str = "Qwen/Qwen3-8B"
    siliconflow_embedding_model: str = "BAAI/bge-large-zh-v1.5"

    max_batch_size: int = 32
    max_chunk_size: int = 256

    knowledge_space: str = "/root/Documents/msds-qa/kb"
