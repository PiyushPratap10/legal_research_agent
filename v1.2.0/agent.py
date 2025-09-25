from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai.base import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from response_generation import get_constitution_response, get_criminal_response, get_civil_response, web_search_tool
import os 
from dotenv import load_dotenv
load_dotenv()
google_api_key=os.getenv("GOOGLE_API_KEY")

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

agent_tools = [
    FunctionTool.from_defaults(
        get_constitution_response,
        name="constitution_research",
        description="Answer legal research questions strictly related to the Indian Constitution. Handles Articles, fundamental rights, directive principles, constitutional duties, amendments, and landmark constitutional cases."
    ),
    FunctionTool.from_defaults(
        get_criminal_response,
        name="criminal_law_research",
        description="Answer legal research questions related to Criminal Law in India, including BNS, BNSS, BSA, IPC, CrPC, offenses (murder, theft, rape, etc.), bail, investigation, and trial procedures."
    ),
    FunctionTool.from_defaults(
        get_civil_response,
        name="civil_law_research",
        description="Answer legal research questions related to Civil Law in India, including contracts, property, family law, torts, civil procedure (CPC), injunctions, damages, remedies, and landmark civil judgments."
    ),
    FunctionTool.from_defaults(
        web_search_tool,
        name="web_legal_search",
        description="Perform a real-time web search for up-to-date legal content (e.g., latest amendments, Supreme Court judgments, or recent cases) when indexed Constitution, Criminal, or Civil data is insufficient."
    )
]


async def legal_agent(query):
    agent = FunctionAgent(
        tools=agent_tools,
        llm=Settings.llm,
        verbose=True,
        system_prompt="""You are a highly specialized Legal Research Agent for Indian law. 
You have access to four expert tools: Constitution Research, Criminal Law Research, Civil Law Research, and Web Legal Search. 
Your job is to carefully choose the correct tool and generate reliable, well-cited answers.

Guidelines:
- Use Constitution Research ONLY for queries about Articles, Fundamental Rights, Directive Principles, constitutional duties, amendments, or landmark constitutional law cases.
- Use Criminal Law Research ONLY for queries about BNS, BNSS, BSA, IPC, CrPC, criminal offenses, rights of accused, bail, investigation, or trial procedures.
- Use Civil Law Research ONLY for queries about contracts, property law, family law, torts, civil procedure (CPC), injunctions, damages, remedies, or landmark civil cases.
- Use Web Legal Search ONLY when:
    1. The user explicitly asks to "search the web" or "find online sources".
    2. Indexed Constitution/Criminal/Civil resources do not provide enough context.
    3. The query requires up-to-date information (latest amendments, judgments, news).

Answering Rules:
- These tools have internal implementations of LLM to generate response so you directly return the output of the tool without any modifications.
- Never hallucinate beyond given sources.
- Be concise, professional, and accurate.
- If multiple tools are relevant, prioritize in this order: Constitution, Criminal, Civil, Web Search."""
    )
    response = await agent.run(query)
    return str(response)


if __name__ == "__main__":
    import asyncio
    query = "What are the fundamental rights guaranteed by the Indian Constitution and how have landmark Supreme Court cases interpreted these rights?"
    response = asyncio.run(legal_agent(query))
    print(response)