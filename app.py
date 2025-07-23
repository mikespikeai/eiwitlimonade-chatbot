import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.readers.sitemap import SitemapReader
from llama_index.llms.openai import OpenAI

# Omgevingsvariabelen
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SITEMAP_URL = os.getenv("SITEMAP_URL")
LANGUAGE = os.getenv("LANGUAGE", "nl")

# Laad en parse de sitemap
documents = SitemapReader().load_data(SITEMAP_URL)
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# Context met GPT-3.5
llm = OpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)

# Bouw index
index = VectorStoreIndex(nodes, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# FastAPI setup
app = FastAPI()

@app.post("/query")
async def query(request: Request):
    body = await request.json()
    vraag = body.get("q", "")
    antwoord = query_engine.query(vraag).response.strip()
    return {"answer": antwoord}

@app.get("/", response_class=HTMLResponse)
async def chat():
    return """
<!DOCTYPE html>
<html><body>
<h3>AI-chat over eiwitlimonade</h3>
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
