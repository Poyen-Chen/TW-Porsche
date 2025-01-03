import os
import json
import glob
import logging 
from uuid import uuid4
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())

class Synthetic_QA_Data(BaseModel):
    user_query: str = Field(description="question asked by the customer")
    positive_answer: str = Field(description="correct answer to the question")
    negative_answer: str = Field(description="wrong answer to the question")


pdf_files = glob.glob("../data/*.pdf")
csv_files = ["../data/porsche_911.csv"]
documents = list()

for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    docs = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=800,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(docs)
    documents.extend(chunks)

for csv_file in csv_files:
    csv_loader = CSVLoader(file_path=csv_file, source_column="link")
    data = csv_loader.load()
    documents.extend(data)

# print(documents)

# uuids = [str(uuid4()) for _ in range(len(documents))]    
# vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings(model="text-embedding-3-large"), ids=uuids)
outputs = dict()
template = """
    You are a well-trained Porsche AI assistant tasked with generating realistic pairs of questions and corresponding positive and negative answers based on given documents. \
    The paired dataset you generate will later be used for the downstream task of fine-tuning the text embedding model for Porsche conversational recommendation system. \
    The question should be something a customer or potential buyer might naturally ask when seeking information about the vehicle model specification. \
    
    
    Given: {chunk}


    Instructions:
    1. Analyze the key topics, facts, and concepts in the given document.
    2. Generate three questions that a user might ask to find the information in this document.
    3. Use natural language and occasionally include typos or colloquialisms to mimic real user behavior in the question. The questions should be related to the scenario when customers ask some queries in Porsche conversational recommendation system. \
    4. Ensure the question is semantically related to the document content WITHOUT directly copying phrases.
    5. Make sure that all of the questions are NOT similar to each other. I.E. All asking questions should cover all topics in the given document.
    6. Hint: \
        Questions like "Could you recommend to me the car model which is the most fuel-efficient?" or "What is the difference between 718 and 911 models?" or "Which one is more fuel-efficient compared to all models?" \
        Positive answers should be concise and precisely based on the document provided.\
        When generating positive answers, you can only retrieve relevant information and rephrase in your own words.
        Negative answers is a hard negative response that only appears to be relevant to the query but turns out to be unrelated and they should be concise. \
        For the negeative answers, it can be made-up and very creative. You can generate whatever you want based on teh questions and use your hallucination. \
    7. Example: \
        An example of quesion-positive/negative-answer pairs like:
        user query: What is the combined CO₂ emissions for the 718 Cayman GTS 4.0?
        positive answer: 230 g/km.
        negative answer: The 718 Cayman GTS 4.0 produces zero emissions because it is fully electric.
    
    
    Output Format:
    Return a JSON object with the following structure:
    ```
    [
        {{
            "user_query": "Generate question",
            "positive_answer": "Generate correct answer based on the question",
            "negative_answer": "Generate wrong answer based on the question"
        }},
        {{
            "user_query": "Generate question",
            "positive_answer": "Generate correct answer based on the question",
            "negative_answer": "Generate wrong answer based on the question"
        }},
        ...
    ]
    ```
    
    The JSON object must contain the following keys:
    - user_query: a string, a random potential buyer asking some specific questions regarding the car models.
    - positive_answer: a string, a relevant answer for the query. 
    - negative_answer: a string, a hard negative answer that only appears to be relevant to the query but is completely wrong.
    
    You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
    Be creative! Think like a curious customer, and generate your 3 questions that would naturally lead to the given document in a semantic search. \
    Ensure your response is a valid JSON object with the required structure.
"""
prompt = ChatPromptTemplate.from_template(template)
json_parser = JsonOutputParser(pydantic_object=Synthetic_QA_Data)
fixing_parser = OutputFixingParser.from_llm(parser=json_parser, llm=ChatOpenAI())

for i in range(len(documents)):
    logger.info(f"Generating sythetic qa pairs for document{i}")
    llm = ChatOpenAI(temperature=1.0, model="gpt-4o-mini")
    chain = prompt | llm | fixing_parser
    try:
        response = chain.invoke({"chunk": documents[i].page_content})
        output_keyname = f"chunk_{i}"
        outputs[output_keyname] = response
    except Exception as e:
        logger.error(f"Error processing document {i}: {e}")
    
with open("syndata.json", "w") as f:
    json.dump(outputs, f)
