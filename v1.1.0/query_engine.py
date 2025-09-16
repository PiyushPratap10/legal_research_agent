import os
from dotenv import load_dotenv
load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY")

from llama_index.core import load_index_from_storage
from llama_index.core.tools import QueryEngineTool
from llama_index.core import StorageContext, Settings, get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai.base import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=google_api_key,
    max_tokens=1000,
    temperature=0.0,
    generation_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category= types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]
    )
    
)
Settings.embed_model=GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=google_api_key,
    embedding_config=types.EmbedContentConfig(
        output_dimensionality=1536,
        task_type="RETRIEVAL_DOCUMENT"
    )
)

def get_summarize_engine():
    client = chromadb.PersistentClient("./v1.1.0/chroma_db")
    collection = client.get_collection("legal-v1.1.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    constitution_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/constitution/vector",vector_store=vector_store)
    constitution_summary_index = load_index_from_storage(storage_context=constitution_storage_ctx)

    criminal_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/criminal/vector",vector_store=vector_store)
    criminal_summary_index=load_index_from_storage(storage_context=criminal_storage_ctx)

    civil_storage_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/civil/vector",vector_store=vector_store)
    civil_summary_index=load_index_from_storage(storage_context=civil_storage_ctx)

    summary_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine= constitution_summary_index.as_query_engine(),
            description="Use this query engine for summarizing information about Indian Constitution, amendments in Indian Constitution, Rights of Indian citizen and constitution related data."
        ),
        QueryEngineTool.from_defaults(
            query_engine=criminal_summary_index.as_query_engine(),
            description="Use this query engine for searching infromation related to Criminal Laws for punishments in India, amendments to criminal and punishments related laws, Bhartiya Nyaya Sanhita, Bhartiya Sakshya Adhiniyam and Bharitya Nagrik Suraksha Sanhita."
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
    client = chromadb.PersistentClient("./v1.1.0/chroma_db")
    collection = client.get_collection("legal-v1.1.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    
    constitution_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/constitution/keyword",vector_store=vector_store)
    constitution_search_index = load_index_from_storage(storage_context=constitution_keyword_ctx)

    criminal_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/criminal/keyword",vector_store=vector_store)
    criminal_search_index=load_index_from_storage(storage_context=criminal_keyword_ctx)

    civil_keyword_ctx = StorageContext.from_defaults(persist_dir="./v1.1.0/indexes/civil/keyword",vector_store=vector_store)
    civil_search_index=load_index_from_storage(storage_context=civil_keyword_ctx)
    summary_engine_tools=[
        QueryEngineTool.from_defaults(
            query_engine= constitution_search_index.as_query_engine(),
            description="Use this query engine for searching information about Indian Constitution, amendments in Indian Constitution, Rights of Indian citizen and constitution related data."
        ),
        QueryEngineTool.from_defaults(
            query_engine=criminal_search_index.as_query_engine(),
            description="Use this query engine for searching infromation related to Criminal Laws for punishments in India, amendments to criminal and punishments related laws, Bhartiya Nyaya Sanhita, Bhartiya Sakshya Adhiniyam and Bharitya Nagrik Suraksha Sanhita."
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
    engine = get_search_engine()

    multi_hop_query_engine = SubQuestionQueryEngine.from_defaults(
        llm=Settings.llm,
        response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.REFINE),
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=engine,
                description="Used for keyword searching and fetching the matching documents."
            )
        ]
    )
    res = multi_hop_query_engine.query("What are the recent amendments or changes to the laws regarding rape and its punishment in India?")
    print(res.response)


