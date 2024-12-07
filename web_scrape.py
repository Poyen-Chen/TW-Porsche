from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

urls = ["https://newsroom.porsche.com/en.html", "https://newsroom.porsche.com/en/products.html"]
loader = AsyncChromiumLoader(urls)
docs = loader.load()
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(docs)
print(docs_transformed)