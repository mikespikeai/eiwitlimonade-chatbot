import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from sitemap_loader import SitemapReader


# Configuratie vanuit omgevingsvariabelen
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SITEMAP_URL = os.getenv("SITEMAP_URL")
LANGUAGE = os.getenv("LANGUAGE", "nl")

# Sitemap inladen
documents = SitemapReader().load_data(SITEMAP_URL)

# Parseren & index bouwen
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex(nodes, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# FastAPI setup
app = FastAPI()

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    vraag = data.get("q", "")
    antwoord = query_engine.query(vraag).response.strip()
    return {"answer": antwoord}

@app.get("/", response_class=HTMLResponse)
async def frontend():
    return """
<!DOCTYPE html>
<html><body>
<h3>AI-chat over jouw site</h3>
<input id="q" placeholder="Stel je vraag..." style="width:80%">
<button onclick="ask()">Vraag</button>
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
