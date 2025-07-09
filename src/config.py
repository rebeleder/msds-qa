from dataclasses import dataclass


@dataclass
class hp:
    ollama_host: str = "http://192.168.215.3:11434"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    ollama_chat_model: str = "qwen3:0.6b"

    supported_ollama_model_family: str = "qwen3"

    knowledge_space: str = "/root/Documents/msds-qa/kb"
