import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from query_engine import keyword_search_tool, summary_tool, google_api_key, web_search_tool
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai.base import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool

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

async def legal_agent(query: str):

    local_agent = FunctionAgent(
        tools=[keyword_search_tool, summary_tool],
        llm=Settings.llm,
        verbose=True,
        system_prompt="""You are a Legal Research AI that provides and summarizes information about Indian legal system, laws, punishments and legal documents.
        If local documents have no infromation related to the query then return - "no relevant results"."""
    )
    local_response = await local_agent.run(query)

    if "no relevant results" in str(local_response).lower():
        web_response = web_search_tool(query)
        return f"⚖️ Local DB did not return relevant info.\n\n🌐 Web Search Results:\n{web_response}"
    
    return str(local_response)



