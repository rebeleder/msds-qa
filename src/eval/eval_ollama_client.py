from src.model import OllamaClient

client = OllamaClient()

chat_model = client.get_chat_model()
embed_model = client.get_embed_model()

query = "Hello, how are you?"

response = chat_model.invoke(query)
embedding = embed_model.embed_query(query)

print("Chat Model Response:", response)
print("Embedding:", embedding)
