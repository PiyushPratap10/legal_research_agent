import os
from dotenv import load_dotenv
load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY")

from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.query_engine import RouterQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=google_api_key)
Settings.embed_model=HuggingFaceEmbedding(model_name="./local_bge_model")

def get_summarize_engine():
    client = chromadb.PersistentClient("./v1.0.0/chroma_db")
    collection = client.get_collection("legal-v1.0.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    constitution_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/constitution/vector",vector_store=vector_store)
    constitution_summary_index = load_index_from_storage(storage_context=constitution_storage_ctx)

    criminal_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/criminal/vector",vector_store=vector_store)
    criminal_summary_index=load_index_from_storage(storage_context=criminal_storage_ctx)

    civil_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/civil/vector",vector_store=vector_store)
    civil_summary_index=load_index_from_storage(storage_context=civil_storage_ctx)

    summary_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine= constitution_summary_index.as_query_engine(),
            description="Use this query engine for summarizing information about Indian Constitution, amendments in Indian Constitution, Rights of Indian citizen and constitution related data."
        ),
        QueryEngineTool.from_defaults(
            query_engine=criminal_summary_index.as_query_engine(),
            description="Use this query engine for summarizing infromation related to Criminal Laws of India, Bhartiya Nyaya Sanhita, Bhartiya Sakshya Adhiniyam and Bharitya Nagrik Suraksha Sanhita."
        ),
        QueryEngineTool.from_defaults(
            query_engine=civil_summary_index.as_query_engine(),
            description="Use this query engine for summarizing information about Civil Laws of India covering Code of Civil Procedure 1908, amendments in Code of Civil Procedure, contract laws, property laws, family laws and corporate laws."
        )
    ]
    summary_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=summary_engine_tools
    )
    return summary_engine

def get_search_engine():
    client = chromadb.PersistentClient("./v1.0.0/chroma_db")
    collection = client.get_collection("legal-v1.0.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    constitution_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/constitution/keyword",vector_store=vector_store)
    constitution_search_index = load_index_from_storage(storage_context=constitution_keyword_ctx)

    criminal_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/criminal/keyword",vector_store=vector_store)
    criminal_search_index=load_index_from_storage(storage_context=criminal_keyword_ctx)

    civil_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.0.0/indexes/civil/keyword",vector_store=vector_store)
    civil_search_index=load_index_from_storage(storage_context=civil_keyword_ctx)

    summary_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine= constitution_search_index.as_query_engine(),
            description="Use this query engine for searching information about Indian Constitution, amendments in Indian Constitution, Rights of Indian citizen and constitution related data."
        ),
        QueryEngineTool.from_defaults(
            query_engine=criminal_search_index.as_query_engine(),
            description="Use this query engine for searching infromation related to Criminal Laws of India, Bhartiya Nyaya Sanhita, Bhartiya Sakshya Adhiniyam and Bharitya Nagrik Suraksha Sanhita."
        ),
        QueryEngineTool.from_defaults(
            query_engine=civil_search_index.as_query_engine(),
            description="Use this query engine for searching information about Civil Laws of India covering Code of Civil Procedure 1908, amendments in Code of Civil Procedure, contract laws, property laws, family laws and corporate laws."
        )
    ]
    search_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=summary_engine_tools
    )
    return search_engine




if __name__=="__main__":
    engine = get_summarize_engine()
    res = engine.query("In Civil laws, Give a summary about Right to Information.")
    print(res.response)


