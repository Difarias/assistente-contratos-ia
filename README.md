# 📄 IA para Análise de Contratos Jurídicos

Sistema de Inteligência Artificial baseado em **RAG (Retrieval-Augmented Generation)** para análise técnica de contratos jurídicos, com foco em identificação de riscos, obrigações e pontos críticos.

---

## 🧠 Sobre o Projeto

Este projeto faz parte de uma iniciativa de pesquisa interdisciplinar desenvolvida em parceria com um projeto de **Iniciação Científica (IC)**, envolvendo alunos de **Ciência da Computação**, **Direito** e **Mestrado em Modelagem Computacional**.

O objetivo é aplicar técnicas modernas de Inteligência Artificial para auxiliar na análise de contratos jurídicos, promovendo integração entre tecnologia e área jurídica.

A aplicação:

- 📥 Lê contratos em PDF  
- ✂️ Divide o texto em partes (chunks)  
- 🧬 Gera embeddings semânticos  
- 🗄️ Armazena em banco vetorial  
- 🔍 Recupera trechos relevantes (MMR)  
- 🤖 Gera respostas com um modelo de linguagem    

---

## ⚙️ Tecnologias Utilizadas

- Python  
- LangChain  
- ChromaDB (banco vetorial)  
- HuggingFace Embeddings (`all-MiniLM-L6-v2`)  
- Ollama (LLM local)  
- Mistral (modelo de linguagem)  

---

## 🏗️ Estrutura do Projeto

```
.
├── base/              # PDFs dos contratos
├── db/                # Banco vetorial (gerado automaticamente)
├── criar_db.py        # Criação do banco vetorial
├── main.py            # Execução da IA
├── .gitignore
└── README.md
```

---

## 🚀 Como Rodar o Projeto

### 1️⃣ Criar ambiente virtual

    python -m venv .venv

### 2️⃣ Ativar ambiente virtual (Windows)

    .venv\Scripts\activate

---

### 3️⃣ Instalar dependências

    pip install langchain
    pip install langchain-community
    pip install langchain-text-splitters
    pip install langchain-chroma
    pip install langchain-huggingface
    pip install langchain-ollama
    pip install chromadb
    pip install python-dotenv
    pip install pypdf
    pip install sentence-transformers

---

### 4️⃣ Instalar o Ollama

Baixe e instale:

https://ollama.com/download

---

### 5️⃣ Baixar o modelo Mistral

    ollama pull mistral

---

### 6️⃣ Preparar os contratos

Crie a pasta:

    base/

Coloque os arquivos PDF dentro dela.

---

### 7️⃣ Criar o banco vetorial

    python criar_db.py

---

### 8️⃣ Executar a IA

    python main.py

---

## 💬 Como Usar

Após rodar o sistema, você pode fazer perguntas como:

- "Quais são as multas previstas no contrato?"
- "Existe cláusula de rescisão?"
- "Quais são os riscos financeiros?"
- "Quais prazos são críticos?"

Digite sua pergunta no terminal e a IA responderá com base no contrato.

---

## 🧠 Como Funciona (RAG)

Fluxo do sistema:

    PDF → Leitura → Chunks → Embeddings → Banco Vetorial → Busca (MMR) → Resposta IA

### 🔍 Estratégia de busca

- Utiliza **MMR (Maximal Marginal Relevance)**  
- Balanceia:
  - relevância  
  - diversidade de contexto  

---

## ⚠️ Observações

- A pasta `db/` é gerada automaticamente e não deve ser versionada  
- O modelo é executado localmente via Ollama  
- A primeira execução pode ser mais lenta devido ao download de modelos  

---
