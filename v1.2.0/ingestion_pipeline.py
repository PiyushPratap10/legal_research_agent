from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, KeywordTableIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.google_genai.base import types
import chromadb
from llama_index.readers.web import SimpleWebPageReader
from website_urls import constitution_urls, criminal_urls

import os
from dotenv import load_dotenv
load_dotenv()

google_api_key=os.getenv("GOOGLE_API_KEY")




def create_vector_index(documents,index_dir):
    client=chromadb.PersistentClient("./v1.2.0/chroma_db")
    collection = client.get_or_create_collection("legal-v1.2.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)

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

    node_parser = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )

    Settings.node_parser = node_parser
    
    nodes=node_parser.get_nodes_from_documents(documents,show_progress=True)

    docstore=SimpleDocumentStore()
    docstore.add_documents(nodes)

    storage_context=StorageContext.from_defaults(vector_store=vector_store,docstore=docstore)
    index=VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )
    index.storage_context.persist(index_dir)

    return index


def create_keyword_table_index(documents,index_dir):
    client=chromadb.PersistentClient("./v1.2.0/chroma_db")
    collection = client.get_or_create_collection("legal-v1.2.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    Settings.llm = GoogleGenAI(
        model="gemini-2.0-flash",
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

    index = KeywordTableIndex.from_documents(documents, storage_context= StorageContext.from_defaults(vector_store=vector_store), show_progress=True)
    index.storage_context.persist(index_dir)

    return index





if __name__=="__main__":
    local_constitution_documents = SimpleDirectoryReader("./data/constitution/", recursive=True).load_data(num_workers=2)
    web_constitution_documents = SimpleWebPageReader(html_to_text=True).load_data(urls=constitution_urls)
    constitution_documents = local_constitution_documents+web_constitution_documents
    constitution_index = create_vector_index(constitution_documents,"./v1.2.0/indexes/constitution/vector/")
    local_criminal_documents = SimpleDirectoryReader("./data/criminal/", recursive=True).load_data(num_workers=2)
    web_criminal_documents = SimpleWebPageReader(html_to_text=True).load_data(urls=criminal_urls)
    criminal_documents = local_criminal_documents + web_criminal_documents
    criminal_index= create_vector_index(criminal_documents,"./v1.2.0/indexes/criminal/vector/")
    civil_documents = SimpleDirectoryReader("./data/civil/", recursive=True).load_data(num_workers=2)
    civil_index= create_vector_index(civil_documents,"./v1.2.0/indexes/civil/vector/")
    sc_documents = SimpleDirectoryReader("./data/sc/", recursive=True).load_data(num_workers=2)
    sc_index= create_vector_index(sc_documents,"./v1.2.0/indexes/sc/vector/")

    civil_kw_idx=create_keyword_table_index(civil_documents,"./v1.2.0/indexes/civil/keyword/")

    cst_kw_index= create_keyword_table_index(constitution_documents,"./v1.2.0/indexes/constitution/keyword/")
    cri_kw_index = create_keyword_table_index(criminal_documents,"./v1.2.0/indexes/criminal/keyword/")