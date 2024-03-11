# genai-pdf-chat
BITS WILP FINAL SEM DISSERTATION PROJECT


## System Requirements

You must have Python 3.9 or later installed. Earlier versions of python may not compile. 

1. Install OLLAMA2

get model
ollama pull llama2
ollama pull mistral

run llama2
ollama run llama2

run mistral
ollama run mistral

list all models
ollama list

ollama rm <model>

## Setup Steps

1. git clone https://github.com/dprasen/genai-pdf-chat.git

2. cd genai-pdf-chat

3. pip install -r requirements.txt

4. place the required pdf files in data folder

5. Ingest Data: python ingest.py

6. Start Chainlit : chainlit run main.py

