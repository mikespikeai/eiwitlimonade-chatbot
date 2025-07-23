import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from sitemap_loader import SitemapReader

# ─────────── SETTINGS ───────────
SITEMAP_URL = "https://eiwitlimonade.nl/sitemap_index.xml"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Zet je OpenAI API-key als omgevingsvariabele."

# ─────────── INDEX ───────────
documents = SitemapReader().load_data(SITEMAP_URL)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY),
    embed_model=OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=5)

# ─────────── FASTAPI ───────────
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, vraag: str = Form(...)):
    try:
        result = query_engine.query(vraag)
        antwoord = result.response
    except Exception as e:
        antwoord = f"Fout: {e}"
    return templates.TemplateResponse("index.html", {"request": request, "vraag": vraag, "antwoord": antwoord})
