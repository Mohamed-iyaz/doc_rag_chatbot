import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load all PDFs
def load_pdf_folder(folder_path):
    text = ""
    for file in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(file))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Step 2: Split text into chunks
def split_text(text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Step 3: Create vector database
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

# Step 4: Ask user questions in terminal
def ask_question(vectorstore):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        docs = vectorstore.similarity_search(query, k=3)
        answer = chain.run(input_documents=docs, question=query)
        print("\nAnswer:", answer)

# Step 5: Main driver
def main():
    folder_path = "pdfs"
    print("Loading and processing PDF files...")
    text = load_pdf_folder(folder_path)
    chunks = split_text(text)
    vectorstore = create_vector_store(chunks)
    print("PDFs processed. Ask your questions.")
    ask_question(vectorstore)

if __name__ == "__main__":
    main()
