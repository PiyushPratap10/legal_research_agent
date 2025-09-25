from llama_index.core import load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai.base import types
from query_preprocessing import query_decomposition
from llama_index.core import StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import re
from llama_index.core.prompts import PromptTemplate
import serpapi
import requests
from bs4 import BeautifulSoup


import os 
from dotenv import load_dotenv
load_dotenv()

#LOADING API KEYS
serp_api_key=os.getenv("SERPAPI_API_KEY")
google_api_key=os.getenv("GOOGLE_API_KEY")

#CONFIGURE LLM AND EMBEDDING MODELS
Settings.llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=google_api_key,
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
))
Settings.embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=google_api_key,
        embedding_config=types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT"
        )
)

client=chromadb.PersistentClient("./v1.2.0/chroma_db")
collection = client.get_collection("legal-v1.2.0")
vector_store = ChromaVectorStore(chroma_collection=collection)


def clean_legal_text(text: str) -> str:
    """
    Cleans raw legal judgement text extracted from nodes.
    Removes citations, footnotes, line numbers, and excessive spacing.
    """
    # 1. Remove SCC/AIR case citations like (1980) 2 SCC 684, AIR 1973 SC 1461
    text = re.sub(r"\(\d{4}\)\s*\d+\s*SCC\s*\d+", " ", text)
    text = re.sub(r"AIR\s*\d{4}\s*SC\s*\d+", " ", text)

    # 2. Remove footnote-style references like [88], [91], [136]
    text = re.sub(r"\[\d+\]", " ", text)

    # 3. Remove inline references like (n 62), (n 73)
    text = re.sub(r"\(n\s*\d+\)", " ", text)

    # 4. Remove standalone numbers (line/page numbers) 
    text = re.sub(r"^\d+\s*$", " ", text, flags=re.MULTILINE)

    # 5. Remove repeated newlines and collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # 6. Trim spaces
    text = text.strip()

    return text

def generate_constitution_context(query):

    stg_ctx_1 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/constitution/vector/",vector_store=vector_store)
    stg_ctx_2 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/constitution/keyword/",vector_store=vector_store)
    vector_index = load_index_from_storage(storage_context=stg_ctx_1)
    keyword_index = load_index_from_storage(storage_context=stg_ctx_2)

    processed_queries= query_decomposition(query,Settings.llm)
    context = []
    if processed_queries:
        for q in processed_queries:
            vector_retriever = vector_index.as_retriever(similarity_top_k=1)
            keyword_retriever = keyword_index.as_retriever(similarity_top_k=1)

            vector_result = vector_retriever.retrieve(q)
            
            result1=vector_result[0].text
            result1 = clean_legal_text(result1)
            if len(result1)>5000:
                result1=result1[:5000]

            
            keyword_result = keyword_retriever.retrieve(q)
            result2 = keyword_result[0].text
            result2= clean_legal_text(result2)
            if len(result2)>5000:
                result2=result2[:5000]

            ctx = result1 +"\n" + result2
            context.append(ctx)

    return context , processed_queries

def generate_criminal_context(query):

    stg_ctx_1 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/criminal/vector/",vector_store=vector_store)
    stg_ctx_2 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/criminal/keyword/",vector_store=vector_store)
    vector_index = load_index_from_storage(storage_context=stg_ctx_1)
    keyword_index = load_index_from_storage(storage_context=stg_ctx_2)

    processed_queries= query_decomposition(query,Settings.llm)
    context = []
    if processed_queries:
        for q in processed_queries:
            vector_retriever = vector_index.as_retriever(similarity_top_k=1)
            keyword_retriever = keyword_index.as_retriever(similarity_top_k=1)

            vector_result = vector_retriever.retrieve(q)
            
            result1=vector_result[0].text
            result1 = clean_legal_text(result1)
            if len(result1)>5000:
                result1=result1[:5000]

            
            keyword_result = keyword_retriever.retrieve(q)
            result2 = keyword_result[0].text
            result2= clean_legal_text(result2)
            if len(result2)>5000:
                result2=result2[:5000]

            ctx = result1 +"\n" + result2
            context.append(ctx)

    return context , processed_queries

def generate_civil_context(query):

    stg_ctx_1 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/civil/vector/",vector_store=vector_store)
    stg_ctx_2 = StorageContext.from_defaults(persist_dir="./v1.2.0/indexes/civil/keyword/",vector_store=vector_store)
    vector_index = load_index_from_storage(storage_context=stg_ctx_1)
    keyword_index = load_index_from_storage(storage_context=stg_ctx_2)

    processed_queries= query_decomposition(query,Settings.llm)
    context = []
    if processed_queries:
        for q in processed_queries:
            vector_retriever = vector_index.as_retriever(similarity_top_k=1)
            keyword_retriever = keyword_index.as_retriever(similarity_top_k=1)

            vector_result = vector_retriever.retrieve(q)
            
            result1=vector_result[0].text
            result1 = clean_legal_text(result1)
            if len(result1)>5000:
                result1=result1[:5000]

            
            keyword_result = keyword_retriever.retrieve(q)
            result2 = keyword_result[0].text
            result2= clean_legal_text(result2)
            if len(result2)>5000:
                result2=result2[:5000]

            ctx = result1 +"\n" + result2
            context.append(ctx)

    return context , processed_queries

# TOOLS FOR RESPONSE GENERATION
RESPONSE_GENERATION_TEMPLATE=PromptTemplate(
    template_var_mappings={"user_query":"query", "sub_queries_list":"sub_queries", "context_list":"context"},
    template="""
You are a Legal Research Agent. Answer the user query based on the context.
- The user query is decomposed into a list of sub queries such that the user query can be answered in-depth.
- The context list contains the respective context for the sub queries.
- Answer the user query if the provided context texts are suitable for generating a correct response.
- Only use information from the context to answer the query.
- If the context is not suitable, return - 'Not enough context to answer your query'.

User Query : {query}

Sub Queries List : {sub_queries}

Context List : {context}

Instructions:
- Provide a clear answer.
- Use bullet points if listing multiple points.
- Always cite relevant article names, case names, or links if available.
- Do NOT hallucinate beyond the given context.

Final Answer:
"""
)

def get_constitution_response(query):
    """
    Answer legal research questions strictly related to the **Indian Constitution**.
    
    This tool retrieves context from a dedicated Constitution index (vector + keyword) 
    and generates detailed answers with citations (Articles, case laws, etc.).

    - Use this tool if the query explicitly asks about:
      * Articles of the Indian Constitution
      * Fundamental rights, directive principles, constitutional duties
      * Constitutional amendments or interpretation of provisions
      * Landmark constitutional law cases (e.g., Kesavananda Bharati, Maneka Gandhi)

    - Example queries:
        * "Explain the scope of Article 21 of the Indian Constitution."
        * "What are the Directive Principles of State Policy?"
        * "What does Article 370 provide?"

    - Do NOT use this tool if:
        * The query is about criminal law, civil law, or general legal principles.
        * The user asks for information outside constitutional provisions.

    Returns:
        str: A detailed legal answer based only on the indexed Constitution documents.
    """

    context,sub_queries = generate_constitution_context(query)

    fmt_prompt = RESPONSE_GENERATION_TEMPLATE.format(
        llm=Settings.llm,
        user_query=query,
        sub_queries_list=sub_queries,
        context_list=context
    )
    # print(fmt_prompt)
    res = Settings.llm.complete(fmt_prompt).text.strip()
    return res

def get_criminal_response(query):
    """
    Answer legal research questions related to **Criminal Law in India**.

    This tool retrieves context from a criminal law index (vector + keyword) 
    and generates detailed answers based on BNS, BNSS, IPC, CrPC, and landmark judgments.

    - Use this tool if the query is about:
      * Bhartiya Nyaya Sanhita (BNS) provisions
      * Bhartiya Nagrik Suraksha Sanhita (BNSS) provisions
      * Bhartiya Sakshya Adhiniyam (BSA) provisions
      * Indian Penal Code (IPC) provisions
      * Code of Criminal Procedure (CrPC)
      * Specific criminal offenses (e.g., murder, theft, defamation, rape)
      * Rights of accused, bail, investigation, or trial procedures

    - Example queries:
        * "Explain the difference between culpable homicide and murder under IPC."
        * "What is the procedure for anticipatory bail under CrPC?"
        * "What are the rights of an arrested person?"

    - Do NOT use this tool if:
        * The query is about constitutional provisions or civil law matters.
        * The query explicitly mentions searching the web.

    Returns:
        str: A legal explanation based only on indexed criminal law resources.
    """

    context,sub_queries = generate_constitution_context(query)

    fmt_prompt = RESPONSE_GENERATION_TEMPLATE.format(
        llm=Settings.llm,
        user_query=query,
        sub_queries_list=sub_queries,
        context_list=context
    )
    res = Settings.llm.complete(fmt_prompt).text.strip()
    return res

def get_civil_response(query):
    """
    Answer legal research questions related to **Civil Law in India**.

    This tool retrieves context from a civil law index (vector + keyword) 
    and generates detailed answers with references to statutes and case laws.

    - Use this tool if the query involves:
      * Contract law, property law, family law, torts
      * Civil procedure (CPC)
      * Injunctions, damages, or remedies in civil matters
      * Landmark civil law judgments

    - Example queries:
        * "What are the essentials of a valid contract under Indian Contract Act?"
        * "Explain the concept of permanent injunction under CPC."
        * "What are the grounds for divorce under Hindu Marriage Act?"

    - Do NOT use this tool if:
        * The query is about criminal law or constitutional law.
        * The user explicitly asks for web results.

    Returns:
        str: A civil law explanation based only on indexed civil law documents.
    """

    context,sub_queries = generate_constitution_context(query)

    fmt_prompt = RESPONSE_GENERATION_TEMPLATE.format(
        llm=Settings.llm,
        user_query=query,
        sub_queries_list=sub_queries,
        context_list=context
    )
    res = Settings.llm.complete(fmt_prompt).text.strip()
    return res

WEB_SEARCH_TEMPLATE=PromptTemplate(
    template_var_mappings={"user_query":"query", "web_context":"context"},
    template="""You are an expert legal research assistant. 
    Use the web content provided below to answer the user query. 
    Only use information from the content. 
    If the content does not contain enough information, say "I could not find enough information in the given sources."

    User Query:
    {query}

    Web Content:
    {context}

    Instructions:
    - Provide a clear and concise answer.
    - Use bullet points if listing multiple points.
    - Always cite relevant article names, case names, or links if available.
    - Do NOT hallucinate beyond the given content.

    Final Answer:"""
)

def web_search(query):
    params = {
        "api_key": serp_api_key,
        "engine": "google",
        "q": query,
        "location": "India",
        "google_domain": "google.co.in",
        "gl": "in",
        "hl": "en",
        "device": "desktop",
        "safe": "active",
        "num": "6"
    }

    search = serpapi.search(params)
    results = search.as_dict()

    cleaned_results = []
    for r in results.get("organic_results", []):
        title = r.get("title")
        link = r.get("link")
        snippet = r.get("snippet")

        try:
            resp = requests.get(link, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p")]
            full_text = "\n".join(paragraphs[:5]) 
        except Exception as e:
            full_text = f"Could not fetch content: {e}"

        cleaned_results.append({
            "title": title,
            "link": link,
            "snippet": snippet,
            "content": full_text
        })

    return cleaned_results

def web_search_tool(query):
    """
    Perform a **real-time web search** and generate an answer using fresh web content. 

    This tool queries Google (via SerpAPI), fetches the top results, scrapes 
    the webpage text, and summarizes them into a legal research answer.  
    It is used when:
      1. The user explicitly asks to "search the web" or "find online sources".
      2. Constitution/Criminal/Civil indexes do not provide enough context.
      3. The query requires **up-to-date information** (e.g., latest amendments, judgments, news).

    - Example queries:
        * "Search the web for the latest Supreme Court judgment on electoral bonds."
        * "Find recent cases about cybercrime in India."
        * "What are the latest changes in the Hindu Succession Act?"
        * (Fallback) If other tools cannot generate a confident answer.

    - Do NOT use this tool if:
        * The query can be fully answered using indexed Constitution/Criminal/Civil data.
        * The user needs purely historical or static legal knowledge.

    Returns:
        str: A concise answer synthesized from recent web content (max ~15,000 chars of context).
    """

    web_results= web_search(query)
    context=[]

    for res in web_results:
        if "could not fetch content" in res['content'].lower():
            continue
        entry = f"Title: {res['title']}\nLink: {res['link']}\nContent: {res['content']}\n"
        context.append(entry)

    context_str = "\n---\n".join(context)
    if len(context_str)>15000:
        context_str=context_str[:15000]

    fmt_prompt = WEB_SEARCH_TEMPLATE.format(
        llm=Settings.llm,
        user_query=query,
        web_context=context_str
    )
    res = Settings.llm.complete(fmt_prompt).text.strip()
    return res
    


if __name__=="__main__":
    query="What are the fundamental rights provided by the Indian Constitution?"
    res=web_search_tool(query)
    print(res)
    
    
