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

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


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

    # Add prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a subject matter expert who provides accurate and eloquent answers to questions.",
            ),
            ("human", "{question}"),
        ]
    )
    return prompt


if __name__ == "__main__":
    create_vector_database()
