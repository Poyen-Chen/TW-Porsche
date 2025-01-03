# Porsche Digital Campus Challenge (Team: TW@Porsche)

[Idea: Recommendation system with fine-tuning text embedding models](https://docs.google.com/presentation/d/1rCkm0kfYHEP12uH-IHF9vGpcYkwnsZtg/edit)

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

## Usage
- `embedding_model.ipynb`: compare fine-tuned model to pretrained model with similarity score as an evaluation index