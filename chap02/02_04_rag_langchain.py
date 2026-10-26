import os
import argparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./data/")
    parser.add_argument("--persist_dir", type=str, default="data_faiss")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")

    # Embeddings Gemini (remplace SentenceTransformerEmbeddings)
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    print(f"Embedding: {embedding.model}")

    if os.path.exists(args.persist_dir):
        # Chargement de l'index FAISS existant
        print(f"Loading FAISS index from {args.persist_dir}")
        vectorstore = FAISS.load_local(
            args.persist_dir,
            embedding,
            allow_dangerous_deserialization=True
        )
        print("done.")

    else:
        # Construction de l'index FAISS
        print(f"Building FAISS index from documents in {args.docs_dir}")

        loader = DirectoryLoader(
            args.docs_dir,
            loader_cls=Docx2txtLoader,
            recursive=True,
            silent_errors=True,
            show_progress=True,
            glob="**/*.docx"  # which files get loaded
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
        )
        frags = text_splitter.split_documents(docs)

        print(f"Populating vector store with {len(docs)} docs in {len(frags)} fragments")
        vectorstore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {args.persist_dir}")
        vectorstore.save_local(args.persist_dir)
        print(f"Saved FAISS index to {args.persist_dir}")

    # LLM Gemini (remplace ChatOpenAI)
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.6
    )

    # Mémoire de conversation (remplace ConversationBufferMemory)
    chat_history = []

    # Prompt avec historique
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant expert.
Réponds uniquement en te basant sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    # Chaîne RAG (remplace ConversationalRetrievalChain)
    retriever = vectorstore.as_retriever()

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # REPL loop
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input == "exit":
            break

        # Ajout à l'historique (remplace memory.chat_memory)
        chat_history.append(HumanMessage(content=user_input))

        response = chain.invoke(user_input)

        chat_history.append(AIMessage(content=response))

        print("AI:", response)


if __name__ == "__main__":
    main()