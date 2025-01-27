# Porsche Digital Campus Challenge (Team: TW@Porsche)

Idea: Recommendation system with fine-tuning text embedding models

[Text Embedding model leaderboard](https://huggingface.co/blog/mteb)

## Requirements

- [Python 3.10 or higher](https://www.python.org/downloads/)
- [LangChain library](https://python.langchain.com/en/latest/index.html)
- [Huggingface API key](https://huggingface.co/login?next=%2Fsettings%2Ftokens)
- [sentence-transformer](https://sbert.net/)
- (Optional) [AWS Bedrock](https://aws.amazon.com/bedrock/?gclid=Cj0KCQiAgdC6BhCgARIsAPWNWH3BUpRDqChA27fLACql5Q6GF8hOnMc7QRxjzNBUevoJRRW2tZxiIKoaAvGcEALw_wcB&trk=a9c05117-53bb-40a3-89b2-a3ee2d23e7d2&sc_channel=ps&ef_id=Cj0KCQiAgdC6BhCgARIsAPWNWH3BUpRDqChA27fLACql5Q6GF8hOnMc7QRxjzNBUevoJRRW2tZxiIKoaAvGcEALw_wcB:G:s&s_kwcid=AL!4422!3!691967569326!e!!g!!amazon%20bedrock!21054971690!157173594137)

## Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Poyen-Chen/TW-Porsche.git
```

#### 2. Create a Python environment

Using `conda`:

```bash
conda env create -f environment.yml
conda activate porsche-challenge
```

#### 3. Set up environment variables

Option 1: open terminal/command window and set up the required enviroment variables

```bash
EXPORT HUGGINGFACEHUB_API_TOKEN="XXXXXXXXX"
```

Option 2: set up the keys in a .env file

First, create a `.env` file in the root directory of the project. Inside the file, add your OpenAI API key and Huggingface API key:

```makefile
HUGGINGFACEHUB_API_TOKEN="your_api_key_here"
```

Save the file and close it. In your Python script or Jupyter notebook, load the `.env` file using the following code:

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

When needed, you can access the `HUGGINGFACEHUB_API_TOKEN ` as an environment variable:

```python
import os
api_key = os.environ['HUGGINGFACEHUB_API_TOKEN']
```

## Dataset:
### From [Porsche Newsroom](https://newsroom.porsche.com/en.html):
Press releases and news articles, Product announcements and specifications, Corporate news and updates, Event coverage, Photo and video galleries
### From [Porsche.com](https://newsroom.porsche.com/en.html):
Vehicle specifications and features, Model information and pricing, Product configurations, Dealership information, Public marketing materials

## Usage
Our fine-tuning dataset can be found in `data` folder:

- `qa_pairs_pos_and_neg.json`: triplet dataset with (anchor, positive, negative)
- `qa_pairs_pos_only.json`: pairs dataset with (anchor, positive)
- `docs/`: contain additional Porsche documents we use other than web scraping to store in our vector database

Due to limited computational resources, most of our work are done in Colab/Kaggle notebook as below:

- `get_data_from_web.ipynb`: web scraping to collect dataset
- `finetuning.ipynb`: fine-tuning embedding models
- `save_finetune_evaluator_results.ipynb`: save the evaluation results into `json` files for visualization in `finetuning_results_evaluation.ipynb`
- `finetuning_results_evaluation.ipynb`: compare fine-tuned model to pretrained model with similarity score as an evaluation index
- `rag_pipeline.ipynb`: integrate into RAG pipeline

Optionally, in `src` folder, there are equivalent `.py` files:

- `web_scrape.py`: scraping from website
- `syndata_generator.py`: generate synthetic dataset
- `config.yaml`: user can input their embedding model and customize their fine-tuning hyperparameters
- `fine_tuning.py`: fine-tuning embedding models
- `rag_baseline.py`: test RAG with baseline pretrained embedding model
- `rag_finetune.py`: test RAG with finetuned embedding model
- `rag_pipeline.py`:  RAG pipeline for future app integration
- `src/aws_bedrock_models`: source codes for generating synthetic dataset using AWS bedrock

The fine-tuned models are stored in `models` folder.

## App demo (Vehicle recommendation system)
Our main goal of this chatbot is let customers who are interested in Porsche vehicles would know more about specific Porsche products. (We used streamlit and adapt the chatbot from this [template](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps).)

Due to computational resource limit, we did not directly integrate RAG pipeline into our UI. We instead store the rsults as `json` and import to our UI. Integration to UI can be done in future work. 
```bash
streamlit run app.py
```
<img width="807" alt="demo" src="https://github.com/Poyen-Chen/TW-Porsche/blob/main/images/demo/demo2.png" />
<img width="807" alt="demo" src="https://github.com/Poyen-Chen/TW-Porsche/blob/main/images/demo/demo3.png" />
<img width="807" alt="demo" src="https://github.com/Poyen-Chen/TW-Porsche/blob/main/images/demo/demo4.png" />
<img width="807" alt="demo" src="https://github.com/Poyen-Chen/TW-Porsche/blob/main/images/demo/demo5.png" />
