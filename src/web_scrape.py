import os
import re
import requests
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import AsyncChromiumLoader, TextLoader, RecursiveUrlLoader
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from bs4 import BeautifulSoup


load_dotenv(find_dotenv())
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")


def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


class WebCrawler:
    def __init__(self, urls, web_scrape_lib="firecrawl"):
        assert web_scrape_lib in ["bs4", "firecrawl", "jina"], \
            "Only beautifulsoup, firecrawl, and jina are supported here."
        self.web_scrape_lib = web_scrape_lib
        self.urls = urls
        self.docs = []
    
    def build_vector_retriever(self):
        if self.web_scrape_lib == "bs4":
            for url in self.urls:
                loader = RecursiveUrlLoader(
                    url,
                    max_depth=5,
                    # use_async=False,
                    extractor=bs4_extractor,
                    # metadata_extractor=None,
                    # exclude_dirs=(),
                    # timeout=10,
                    # check_response_status=True,
                    # continue_on_failure=True,
                    # prevent_outside=True,
                    # base_url=None,
                    # ...
                )
                self.docs = loader.load()
                
        elif self.web_scrape_lib == "firecrawl":
            loader = FireCrawlLoader(api_key=FIRECRAWL_API_KEY, url=self.urls, mode="crawl")
            self.docs = loader.load()
            
        else:
            for url in self.urls:
                loader = RecursiveUrlLoader(
                    url,
                    max_depth=5,
                    # use_async=False,
                    # extractor=None,
                    # metadata_extractor=None,
                    # exclude_dirs=(),
                    # timeout=10,
                    # check_response_status=True,
                    # continue_on_failure=True,
                    # prevent_outside=True,
                    # base_url=None,
                    # ...
                )
                raw_docs = loader.load()
                for i in range(len(raw_docs)):
                    src_url = raw_docs[i].metadata['source']
                    jina_url = f"https://r.jina.ai/{src_url}"
                    try:
                        response = requests.get(jina_url)
                        if response.status_code == 200:
                            document = Document(
                                page_content=response,
                                metadata={"source": src_url}
                            )
                            self.docs.extend(document)
                        else:
                            print(f"Failed to fetch data. Status code: {response.status_code}")
                            return None
                    except requests.exceptions.RequestException as e:
                        print(f"An error occurred: {e}")
                        return None
            
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        if len(self.docs) == 0:
            raise Exception("Sorry, no documents found during web scraping. Please check your source urls.")
        docs = text_splitter.split_documents(self.docs)
        # embeddings = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
        # you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
        embeddings = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )
        db = FAISS.from_documents(docs, embeddings)

        return db.as_retriever()

    
if __name__ == "__main__":
    urls = ["https://www.porsche.com/international/models/", "https://www.porsche.com/central-eastern-europe/en/models/",
            "https://newsroom.porsche.com/en/products.html", "https://www.porsche.com/international/accessoriesandservice/classic/models/"]
    webcrawler = WebCrawler(urls, web_scrape_lib="jina") 
    vector_store = webcrawler.build_vector_retriever()
    vector_store.save_local("faiss_index")
