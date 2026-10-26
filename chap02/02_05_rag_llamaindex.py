import os
import argparse
import docx2txt
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./data/")
    parser.add_argument("--persist_dir", type=str, default="data_llamaindex")
    args = parser.parse_args()

    print(f"Using data dir {args.docs_dir}")
    print(f"Using index path {args.persist_dir}")

    # LLM Gemini
    Settings.llm = GoogleGenAI(
        model=os.getenv("GEMINI_MODEL"),
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Embeddings locaux SentenceTransformer
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="all-mpnet-base-v2"
    )
    print(f"Embedding: {Settings.embed_model.model_name}")

    if os.path.exists(args.persist_dir):
        # Chargement de l'index existant
        print(f"Loading index from {args.persist_dir}")
        storage_context = StorageContext.from_defaults(
            persist_dir=args.persist_dir
        )
        index = load_index_from_storage(storage_context)
        print("done.")

    else:
        print(f"Building index from documents in {args.docs_dir}")

        # Lecture manuelle de TOUS les .docx
        documents = []
        for filename in os.listdir(args.docs_dir):
            if filename.endswith(".docx"):
                filepath = os.path.join(args.docs_dir, filename)
                print(f"  📖 Lecture de {filename}...")
                text = docx2txt.process(filepath)
                documents.append(Document(
                    text=text,
                    metadata={"filename": filename}
                ))

        print(f"📄 {len(documents)} document(s) chargé(s)")

        # Découpage en chunks
        splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=64
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"✂️  {len(nodes)} chunk(s) créé(s)")

        # Création de l'index
        print(f"🔄 Embedding de {len(nodes)} chunks en local...")
        index = VectorStoreIndex(
            nodes=nodes,
            show_progress=True,
            transformations=[]
        )

        # Sauvegarde
        index.storage_context.persist(persist_dir=args.persist_dir)
        print(f"✅ Index sauvegardé dans {args.persist_dir}")

    # Moteur de conversation avec mémoire
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        verbose=True,
        system_prompt=(
            "Tu es un assistant concis et précis. "
            "Réponds uniquement à la question posée en 2-3 phrases maximum. "
            "Ne donne pas d'informations supplémentaires non demandées. "
            "Base-toi uniquement sur le contenu des documents."
        )
    )

    # REPL loop
    while True:
        user_input = input("Ask a question. Type 'exit' to quit.\n>")
        if user_input == "exit":
            break

        try:
            response = chat_engine.chat(user_input)
            print("AI:", response)
        except Exception as e:
            if "429" in str(e):
                print("  Quota dépassé. Attends quelques minutes et réessaie.")
            else:
                raise e


if __name__ == "__main__":
    main()