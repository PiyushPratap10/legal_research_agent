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
from ddgs import DDGS

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


def keyword_search_tool(query):
    """
    Tool Name: Legal Keyword Search Engine

    Purpose:
    This tool performs **exact keyword-based search** over the legal corpus. 
    It is most effective when the user asks about:
    - Specific legal terms, phrases, or sections (e.g., "Article 370", "Section 144 CrPC")
    - Precise wording of statutes, amendments, or case laws
    - Citations, act names, or specific clauses
    - Direct references to articles, sections, rules, or amendments

    When to Use:
    - Use when the query is **narrow, fact-based, or lookup-oriented**.
    - Do NOT use this tool for broad, explanatory, or analytical questions.
    - Prefer this when the user explicitly mentions a legal reference (e.g., "Article", "Section", "Amendment").

    Output:
    Returns the **closest matching documents** containing the exact legal references or phrases,
    which can then be refined or contextualized by the summarization/vector engines.
    """
    
    engine = get_search_engine()
    multi_hop_query_engine = SubQuestionQueryEngine.from_defaults(
        llm=Settings.llm,
        response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.ACCUMULATE),
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=engine,
                description=("Keyword-based legal search engine. Best for retrieving exact legal references "
                    "(Articles, Sections, Amendments, case names). Optimized for precise lookups, "
                    "not for explanations or summaries.")
            )
        ]
    )
    res=multi_hop_query_engine.query(query)
    return str(res.response)

def summary_tool(query):
    """ Tool Name: Legal Summarization Engine

    Purpose:
    This tool performs **high-level summarization and synthesis** across multiple legal documents. 
    It is designed to provide **coherent, structured, and comprehensive answers** rather than 
    exact keyword matches.

    When to Use:
    - Use when the user asks for **explanations, overviews, or summaries** of legal topics.
    - Best for broad or analytical queries such as:
        • "Summarize the Fundamental Rights in the Constitution"
        • "Explain the differences between civil and criminal law"
        • "Provide an overview of amendments related to freedom of speech"
    - Do NOT use this tool for pinpoint lookups of specific articles, sections, or phrases 
      (use the Keyword Search Engine for that).

    Output:
    Returns a **refined, multi-document summary** synthesized into natural language, 
    making complex legal information easier to understand."""

    engine = get_summarize_engine()
    multi_hop_query_engine = SubQuestionQueryEngine.from_defaults(
        llm=Settings.llm,
        response_synthesizer=get_response_synthesizer(response_mode=ResponseMode.TREE_SUMMARIZE),
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=engine,
                description=(
                    "Summarization engine for broad, explanatory, or analytical legal queries. "
                    "Best for generating overviews, comparisons, or topic-based summaries across documents."
                )
            )
        ]
    )
    res=multi_hop_query_engine.query(query)
    return str(res.response)

def web_search_tool(query):
    """ Use this tool ONLY when the existing legal document corpus does not 
    contain relevant info. This searches the web for the latest information. ONLY search for queries related to Legal domains."""

    engine=DDGS()
    res=""
    try:
        results=engine.text(query,max_results=3)
    except Exception as e:
        print(e)
        return "Error searching the web!"
    
    for result in results:
        res+=result["body"]
    return res if res!="" else "No information found!"




