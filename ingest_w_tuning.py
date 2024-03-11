import os
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")

# Simple caching mechanism using a dictionary
cache = {}

# Fine-tune the model on new data
def fine_tune_model():
    # Implement fine-tuning process here
    print("Fine-tuning the model...")

# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown, and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.
    """
    # Initialize loaders for different file types
    pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)    
    loaded_documents = pdf_loader.load()

    # Load documents and split them into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    chunked_documents = [text_splitter.split_documents(documents) for documents in loaded_documents]

    # Initialize Ollama Embeddings
    ollama_embeddings = OllamaEmbeddings(model="mistral")

    # Create and persist a Chroma vector database from the chunked documents
    vector_database = Chroma.from_documents(
        documents=[chunk for chunks in chunked_documents for chunk in chunks],
        embedding=ollama_embeddings,
        persist_directory=DB_DIR,
    )

    vector_database.persist()

# Process question and trigger fine-tuning
def process_question(question):
    if question in cache:
        return cache[question]

    fine_tune_model()
    # Process the question and generate a response
    print("Processing question:", question)
    response = "Response to question: " + question

    # Cache the response
    cache[question] = response
    return response

if __name__ == "__main__":
    create_vector_database()

    # Example of processing questions
    questions = ["What is the capital of France?", "Who is the president of the USA?"]
    for question in questions:
        response = process_question(question)
        print("Response:", response)
