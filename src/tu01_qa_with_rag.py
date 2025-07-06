# https://haystack.deepset.ai/tutorials/27_first_rag_pipeline
import os
import sys

from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.utils import ComponentDevice
from haystack.dataclasses import ChatMessage

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator


if "OLLAMA_URL" not in os.environ:
    print("OLLAMA_URL not found in environment. Exiting.")
    sys.exit(1)

# use 2nd RTX A4000
device = ComponentDevice.from_str("cuda:1")

# init Transformers
print("init Transformers")
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
doc_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, device=device)
doc_embedder.warm_up()
text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, device=device)

# init DocumentStore
print("init DocumentStore")
document_store = InMemoryDocumentStore()
retriever = InMemoryEmbeddingRetriever(document_store)

# download Dataset & index Documents
print("index Documents")
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# Chat Generator
chat_generator=OllamaChatGenerator(model="qwen3:14b", url = os.environ["OLLAMA_URL"])

# Prompt Builder
template = [
    ChatMessage.from_user(
"""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]
prompt_builder = ChatPromptBuilder(template=template)

# init Pipeline
print("init Pipeline")
basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# connect the rag components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# run the pipeline
try:
    while True:
        print()
        question = input("\nAsk a question (CTRL-C to exit): ")
        response = basic_rag_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        print("\nAnswer:", response["llm"]["replies"][0].text)
except KeyboardInterrupt:
    print("\nExiting.")
    sys.exit(1)
