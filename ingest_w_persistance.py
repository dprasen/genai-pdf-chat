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
import chainlit as cl

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

    with ThreadPoolExecutor() as executor:
        chunked_documents = list(executor.map(process_document, loaded_documents))

    ollama_embeddings = OllamaEmbeddings(model="mistral")

    vector_database.persist()


def preprocess_question(question):
    return question.lower().strip()


def store_qa_pair(question, answer):
    cache[preprocess_question(question)] = answer
    # Store question-answer pair in ChromaDB
    vector_database.put(preprocess_question(question), answer)


def load_model(temperature=1.0):
    pass


def fine_tune_model(data):
    pass


def qa_bot(temperature=1.0):
    if temperature in cache:
        return cache[temperature]

    llm = load_model(temperature)
    DB_PATH = DB_DIR
    vectorstore = Chroma(
        persist_directory=DB_PATH, embedding_function=OllamaEmbeddings(model="mistral")
    )

    qa = retrieval_qa_chain(llm, vectorstore)

    cache[temperature] = qa
    return qa


async def start():
    fine_tune_data = ...  # Load and preprocess additional data
    fine_tune_model(fine_tune_data)
    load_model(temperature=1.0)

    chain = qa_bot(temperature=1.0)

    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (mistral model) and LangChain."
    )
    await welcome_message.update()

    cl.user_session.set("chain", chain)


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True,
    )
    return qa_chain


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()


if __name__ == "__main__":
    create_vector_database()
