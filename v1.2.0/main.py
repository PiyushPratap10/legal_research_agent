import chromadb
from fastapi import FastAPI, Request
from fastapi_socketio import SocketManager
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from agent import legal_agent
import globals

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    client=chromadb.PersistentClient("chroma_db")
    collection = client.get_collection("legal-v1.2.0")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    #Constitution Indexes
    const_stg_ctx_1 = StorageContext.from_defaults(persist_dir="indexes/constitution/vector/",vector_store=vector_store)
    const_stg_ctx_2 = StorageContext.from_defaults(persist_dir="indexes/constitution/keyword/",vector_store=vector_store)
    const_vector_index = load_index_from_storage(storage_context=const_stg_ctx_1)
    const_keyword_index = load_index_from_storage(storage_context=const_stg_ctx_2)
    globals.const_vector_index = const_vector_index
    globals.const_keyword_index = const_keyword_index

    #Criminal Law Indexes
    cri_stg_ctx_1 = StorageContext.from_defaults(persist_dir="indexes/criminal/vector/",vector_store=vector_store)
    cri_stg_ctx_2 = StorageContext.from_defaults(persist_dir="indexes/criminal/keyword/",vector_store=vector_store)
    cri_vector_index = load_index_from_storage(storage_context=cri_stg_ctx_1)
    cri_keyword_index = load_index_from_storage(storage_context=cri_stg_ctx_2)
    globals.cri_vector_index = cri_vector_index
    globals.cri_keyword_index = cri_keyword_index

    #Civil Law Indexes
    civ_stg_ctx_1 = StorageContext.from_defaults(persist_dir="indexes/civil/vector/",vector_store=vector_store)
    civ_stg_ctx_2 = StorageContext.from_defaults(persist_dir="indexes/civil/keyword/",vector_store=vector_store)
    civ_vector_index = load_index_from_storage(storage_context=civ_stg_ctx_1)
    civ_keyword_index = load_index_from_storage(storage_context=civ_stg_ctx_2)
    globals.civ_vector_index = civ_vector_index
    globals.civ_keyword_index = civ_keyword_index

    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

socket_manager = SocketManager(app=app, mount_location="/socket.io", cors_allowed_origins=["*"])
templates = Jinja2Templates(directory="templates")



@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@socket_manager.on("message")
async def handle_message(sid, message):
    print(f"Message from {sid}: {message}")
    response = await legal_agent(message,sid)
    await socket_manager.emit("response", response, to=sid)

