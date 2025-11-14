import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_all_pdfs(data_dir: str):
    """
    Recursively loads all PDF files from a given directory.
    Adds source metadata and prints total documents loaded.
    Returns a list of LangChain Document objects.
    """
    all_documents = []

    # 1️⃣ Find all PDF files recursively
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                print(f"📄 Loading: {file_path}")

                # 2️⃣ Load the PDF
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()

                # 3️⃣ Add source info in metadata
                for doc in pdf_docs:
                    doc.metadata["source"] = file_path

                # 4️⃣ Append to final list
                all_documents.extend(pdf_docs)

    # 5️⃣ Split text into chunks (recommended for RAG)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = text_splitter.split_documents(all_documents)

    print(f"\n✅ Total documents loaded: {len(split_docs)}")
    return split_docs
