import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import OpenAI

from sitemap_loader import SitemapReader

# Laad documenten uit je sitemap
SITEMAP_URL = "https://eiwitlimonade.nl/sitemap_index.xml"
documents = SitemapReader().load_data(SITEMAP_URL)

# Instellingen voor OpenAI + indexering
llm = OpenAI(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Start FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query", response_class=HTMLResponse)
async def query(request: Request, vraag: str = Form(...)):
    try:
        response = query_engine.query(vraag)
        antwoord = response.response.strip()
    except Exception as e:
        antwoord = f"Er ging iets mis: {str(e)}"
    return templates.TemplateResponse("index.html", {"request": request, "vraag": vraag, "antwoord": antwoord})
