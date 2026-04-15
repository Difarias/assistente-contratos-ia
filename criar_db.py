from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings     
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

PASTA_BASE = "base"
CAMINHO_DB = "db"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

def carregar_documentos():
    if not os.path.exists(PASTA_BASE):
        raise Exception(f"Pasta '{PASTA_BASE}' não encontrada.")

    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()

    if not documentos:
        raise Exception("Nenhum PDF encontrado na pasta 'base'.")

    print(f"{len(documentos)} documentos carregados.")
    return documentos

def dividir_chunks(documentos):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        add_start_index=True
    )

    chunks = separador.split_documents(documentos)

    print(f"{len(chunks)} chunks criados.")
    return chunks

import shutil
import os

def vetorizar_chunks(chunks):
    if os.path.exists("db"):
        shutil.rmtree("db")  # apaga banco antigo

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="db"
    )

    print("✅ Banco vetorial recriado com sucesso!")

if __name__ == "__main__":
    criar_db()
    