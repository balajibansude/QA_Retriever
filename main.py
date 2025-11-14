import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1️⃣ Load PDF
pdf_path = "data/sample.pdf"  # put your PDF file in data/ folder
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2️⃣ Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 3️⃣ Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 4️⃣ Store embeddings in FAISS
vectorstore = FAISS.from_documents(texts, embeddings)

# 5️⃣ Create retriever and chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
    retriever=retriever,
    chain_type="stuff",
)

# 6️⃣ Ask a question
query = "Summarize the main idea of the document."
response = qa_chain.run(query)
print("🧠 Answer:", response)
