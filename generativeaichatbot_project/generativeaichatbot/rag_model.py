from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from pymilvus import Collection

class RagModel:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
        self.collection = Collection("chatbot_collection")

    def generate_response(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        # Generate embeddings and store in Milvus
        embedding = self.model.get_input_embeddings()(inputs['input_ids'])
        entities = [
            {"name": "embedding", "values": embedding.detach().numpy().tolist()},
            {"name": "text", "values": [input_text]}
        ]
        self.collection.insert(entities)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text[0]
