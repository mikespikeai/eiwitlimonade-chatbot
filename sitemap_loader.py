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
                page = requests.get(url)
                text = BeautifulSoup(page.text, "html.parser").get_text()
                documents.append(Document(text=text, metadata={"url": url}))
            except Exception as e:
                print(f"Fout bij {url}: {e}")
        return documents
