from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

CAMINHO_DB = "db"

print("🔄 Inicializando Inteligência Contratual...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embeddings)

# ESTRATÉGIA DE BUSCA ROBUSTA (MMR)
# fetch_k=25 analisa 25 trechos e escolhe os 10 mais relevantes e distintos entre si
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 25,
        "lambda_mult": 0.5  # Balanço entre relevância (1.0) e diversidade (0.0)
    }
)

# Modelo Mistral configurado para precisão máxima
modelo = ChatOllama(model="mistral", temperature=0.1)

# PROMPT 
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Você é um Especialista em Auditoria de Contratos e Gestão de Riscos Jurídicos.
    Sua missão é realizar uma análise técnica, cética e extremamente precisa dos trechos fornecidos.

    DIRETRIZES DE RESPOSTA:
    1. IDIOMA: Responda exclusivamente em PORTUGUÊS (PT-BR).
    2. CITAÇÃO: Sempre que mencionar uma obrigação, multa ou prazo, cite brevemente o contexto do trecho (ex: "Conforme a cláusula de rescisão...").
    3. PONTOS DE ATENÇÃO: Foque em:
       - Riscos financeiros (multas, juros, correções).
       - Prazos críticos e condições de renovação.
       - Obrigações ocultas ou desequilibradas entre as partes.
       - Condições de saída e distrato.
    4. CETICISMO: Não assuma fatos que não estão nos trechos. Se um dado for ambíguo, aponte como "Risco de Interpretação".
    5. FORMATO: Use negrito para termos chave e listas (bullet points) para clareza.

    Base de conhecimento (trechos do contrato):
    {base_conhecimento}"""),
    MessagesPlaceholder(variable_name="historico"),
    ("human", "{pergunta}")
])

historico_chat = []

def perguntar():
    global historico_chat
    pergunta_usuario = input("\n -> Usuário: ")

    if pergunta_usuario.lower() in ["sair", "exit", "quit"]:
        print("Encerrando sistema...")
        exit()

    print("Realizando procura no documento...")

    # Recuperação otimizada
    docs = retriever.invoke(pergunta_usuario)
    conteudos = [d.page_content for d in docs]
    base_conhecimento = "\n\n---\n\n".join(conteudos)

    # Execução da Chain
    chain = prompt_template | modelo
    
    resposta_objeto = chain.invoke({
        "pergunta": pergunta_usuario,
        "base_conhecimento": base_conhecimento,
        "historico": historico_chat
    })

    resposta_texto = resposta_objeto.content
    print("\nResp da IA:\n", resposta_texto)

    # Gerenciamento de Memória
    historico_chat.append(HumanMessage(content=pergunta_usuario))
    historico_chat.append(AIMessage(content=resposta_texto))
    
    if len(historico_chat) > 6:
        historico_chat = historico_chat[-6:]

def main():
    print("=" * 60)
    print("IA DE CONTRATO")
    print("=" * 60)
    
    while True:
        try:
            perguntar()
        except Exception as e:
            print(f"Erro crítico no processamento: {e}")

if __name__ == "__main__":
    main()