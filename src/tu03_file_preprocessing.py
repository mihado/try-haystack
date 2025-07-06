from pathlib import Path

from haystack import Pipeline
from haystack.utils import ComponentDevice

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# use 2nd RTX A4000
device = ComponentDevice.from_str("cuda:1")

# init Transformers
print("init Transformers")
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, device=device)
document_embedder.warm_up()

# init DocumentStore
print("init DocumentStore")
document_store = InMemoryDocumentStore()

document_cleaner = DocumentCleaner()
document_joiner = DocumentJoiner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
document_writer = DocumentWriter(document_store)

# file Converters
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()

# init Pipeline
print("init Pipeline")
preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

# connect the rag components to each other
preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")


# process documents
print("preprocessing_pipeline.run")
resources_dir="../res/Tutorial-30"
preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(resources_dir).glob("**/*"))}})
