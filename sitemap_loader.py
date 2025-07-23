import requests
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
from typing import List

class SitemapReader:
    def load_data(self, sitemap_url: str) -> List[Document]:
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        
        documents = []
        for url in urls:
            try:
                page = requests.get(url, timeout=5)
                if page.status_code == 200:
                    documents.append(Document(text=page.text, metadata={"source": url}))
            except Exception as e:
                print(f"Fout bij ophalen van {url}: {e}")
        return documents
