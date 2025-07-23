import requests
from bs4 import BeautifulSoup
from llama_index.core.schema import Document

class SitemapReader:
    def load_data(self, sitemap_url):
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]

        documents = []
        for url in urls:
            try:
                page = requests.get(url, timeout=10)
                soup = BeautifulSoup(page.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                documents.append(Document(text=text, metadata={"url": url}))
            except Exception as e:
                print(f"Fout bij laden van {url}: {e}")
        return documents
