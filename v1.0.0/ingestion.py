import os
from dotenv import load_dotenv
load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY")

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, DocumentSummaryIndex, KeywordTableIndex, get_response_synthesizer, VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.google_genai import GoogleGenAI
from website_urls import constitution_urls, criminal_urls
from llama_index.core.response_synthesizers import ResponseMode

Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=google_api_key)
Settings.embed_model=HuggingFaceEmbedding(model_name="./local_bge_model")


def ingestion_pipeline():
    client = chromadb.PersistentClient("./v1.0.0/chroma_db")
    chroma_collection = client.get_or_create_collection("legal-v1.0.0")

    web_constitution_documents = SimpleWebPageReader(html_to_text=True).load_data(urls=constitution_urls)
    web_criminal_documents = SimpleWebPageReader(html_to_text=True).load_data(urls=criminal_urls)
    local_constitution_documents = SimpleDirectoryReader("./data/constitution/", recursive=True).load_data(num_workers=2)
    local_criminal_documents = SimpleDirectoryReader("./data/criminal/", recursive=True).load_data(num_workers=2)
    civil_documents = SimpleDirectoryReader("./data/civil/", recursive=True).load_data(num_workers=2)

    constitution_documents = local_constitution_documents + web_constitution_documents
    criminal_documents = local_criminal_documents + web_criminal_documents

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    constitution_keyword_index = KeywordTableIndex.from_documents(
        constitution_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True
    )
    constitution_keyword_index.storage_context.persist("./v1.0.0/indexes/constitution/keyword/")

    criminal_keyword_index = KeywordTableIndex.from_documents(
        criminal_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True
    )
    criminal_keyword_index.storage_context.persist("./v1.0.0/indexes/criminal/keyword/")

    civil_keyword_index = KeywordTableIndex.from_documents(
        civil_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True
    )
    civil_keyword_index.storage_context.persist("./v1.0.0/indexes/civil/keyword/")

    constitution_vector_store_index = VectorStoreIndex.from_documents(
        constitution_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True,
    )
    constitution_vector_store_index.storage_context.persist("./v1.0.0/indexes/constitution/vector/")

    criminal_vector_store_index = VectorStoreIndex.from_documents(
        criminal_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True,
    )
    criminal_vector_store_index.storage_context.persist("./v1.0.0/indexes/criminal/vector/")

    civil_vector_store_index = VectorStoreIndex.from_documents(
        civil_documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True,
    )
    civil_vector_store_index.storage_context.persist("./v1.0.0/indexes/civil/vector/")


if __name__=="__main__":
    ingestion_pipeline()
    

