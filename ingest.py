import os
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ChatPromptTemplate

from main import retrieval_qa_chain

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Simple caching mechanism using a dictionary
cache = {}


def process_document(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = text_splitter.split_documents([document])
    return chunked_documents[0]


def create_vector_database():
    pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    loaded_documents = pdf_loader.load()

    # Optimize data loading using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        chunked_documents = list(executor.map(process_document, loaded_documents))

    ollama_embeddings = OllamaEmbeddings(model="mistral")

    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()


def load_model(temperature=1.0):
    # Here you can initialize and configure your model as needed
    # For fine-tuning, load a pre-trained model and fine-tune it on additional data
    pass


def fine_tune_model(data):
    # Fine-tune the pre-trained model on the additional data
    pass


def qa_bot(temperature=1.0):
    # Check if the response is cached
    if temperature in cache:
        return cache[temperature]

    # Initialize and warm up the model during system startup
    llm = load_model(temperature)

    DB_PATH = DB_DIR
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="mistral")
    )

    qa = retrieval_qa_chain(llm, vectorstore)

    # Cache the response
    cache[temperature] = qa
    return qa


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    # Fine-tune the model on additional data
    fine_tune_data = ...  # Load and preprocess additional data
    fine_tune_model(fine_tune_data)

    # Warm up the model during system startup
    load_model(temperature=1.0)

    # Initialize the bot
    chain = qa_bot(temperature=1.0)

    # Send welcome message
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (mistral model) and LangChain."
    )
    await welcome_message.update()

    # Store the bot instance in the user's session
    cl.user_session.set("chain", chain)


if __name__ == "__main__":
    create_vector_database()
