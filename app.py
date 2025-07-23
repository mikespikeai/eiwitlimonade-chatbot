import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.readers.sitemap import SitemapReader
from openai import OpenAI

import asyncio

SITEMAP_URL = os.getenv("SITEMAP_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
LANG = os.getenv("LANGUAGE", "nl")

# Crawling: haal pagina’s op via sitemap
from llama_index.readers.sitemap import SitemapReader
documents = SitemapReader().load_data(SITEMAP_URL)
index = VectorStoreIndex.from_documents(documents,
    service_context=ServiceContext.from_defaults(
        chunk_size=512,
        llm=OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo", temperature=0),
    ))

app = FastAPI()

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    query_str = body.get("q", "")
    resp = index.query(query_str)
    answer = resp.response.strip()
    if resp.source_nodes is None:
        answer = "Sorry, ik heb daar geen informatie over."
    return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
async def chat():
    return """
<!DOCTYPE html>
<html><body>
<h3>AI-chat over jouw site</h3>
<input id="q" placeholder="Stel je vraag..." style="width:80%"><button onclick="ask()">Vraag</button>
<pre id="out"></pre>
<script>
async function ask() {
  let q = document.getElementById("q").value
  let res = await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({q})
  })
  let j = await res.json()
  document.getElementById("out").innerText = j.answer
}
</script>
</body></html>
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
