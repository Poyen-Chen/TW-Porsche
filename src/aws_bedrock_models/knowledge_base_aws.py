from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from botocore.client import Config
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_aws import ChatBedrock
import pprint
from dotenv import load_dotenv, find_dotenv
import boto3
import json

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

load_dotenv(find_dotenv())
pp = pprint.PrettyPrinter(indent=2)
session = boto3.session.Session()
region = session.region_name   # use can you the region of your choice.
bedrock_config = Config(
    connect_timeout=120, read_timeout=120, retries={'max_attempts': 0}
)
bedrock_client = boto3.client('bedrock-runtime', region_name = region)

# Define the LLM
model_kwargs_claude = {"temperature": 0.1, "top_k": 10}
llm = ChatBedrock(model_id="anthropic.claude-3-5-haiku-20241022-v1:0", model_kwargs=model_kwargs_claude, client=bedrock_client)

# Configure the retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="BMBQNIG70W",
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4,
            "overrideSearchType": "HYBRID",
        }
    }
)

# Prompt template
PROMPT_TEMPLATE = """
Human: You are a Porsche AI assistant, and provides answers to questions by using fact based information. 
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and informative but yet concise.

Assistant:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# RAG
#query = "What are the main features of 911 Carrera Models?"
with open("questions.json", 'r') as f:
    questions = json.load(f)
qa_pairs = {}
for query_idx, query in questions.items():
    #combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(query)
    #pp.pprint(response)
    qa_pairs[query_idx] = query
    idx = query_idx.removeprefix("question_")
    qa_pairs[f"pos_ans_{idx}"] = response
    with open('qa_pairs.json', 'w') as f:
        json.dump(qa_pairs, f)
print("All queries completed and saved!")
