import json
import argparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def embed_data(input_json, output_dir):
    print(f"Loading data from {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    # You asked a brilliant question about chunk size.
    # Yes, for financial reports, 512 is often too small! Small chunks lose the surrounding context
    # (like the preamble to a table or the paragraph explaining a metric).
    # If the LLM only sees "Revenue increased by 15%", it won't know WHICH revenue or for WHICH year
    # if the context was left in the previous chunk.
    # 
    # Let's increase it to 1000 tokens with a 200 token overlap to ensure ideas aren't cut in half.
    # The Llama 3.3 70B model via Groq has a massive 8k+ context window anyway, so passing 
    # 3-5 large chunks (3000-5000 tokens total) is perfectly fine and yields MUCH better answers.
    print("Setting up text splitter (1000 chunk size, 200 overlap)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )

    documents = []
    
    for page in pages:
        page_num = page.get("page", "unknown")
        content_type = page.get("type", "unknown")
        
        # 1. Process regular/OCR text
        text = page.get("text", "")
        if text.strip():
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "type": content_type,
                        "chunk_index": i
                    }
                )
                documents.append(doc)
                
        # 2. Process tables separately (preserve their structure!)
        tables = page.get("tables", [])
        for i, table_text in enumerate(tables):
            if table_text.strip():
                # We generally don't split tables unless they are massive.
                # A single table chunk is better for the LLM to read.
                doc = Document(
                    page_content=table_text,
                    metadata={
                        "page": page_num,
                        "type": "table",
                        "table_index": i
                    }
                )
                documents.append(doc)

    print(f"Generated {len(documents)} chunks (documents).")
    
    print("Initializing embedding model (BAAI/bge-small-en-v1.5)...")
    # bge-small is highly ranked on MTEB leaderboards, fast, and runs locally on CPU perfectly.
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Creating FAISS vector index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "faiss_index")
    vectorstore.save_local(index_path)
    print(f"Index successfully saved to {index_path} !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk extracted PDF JSON and embed into FAISS")
    parser.add_argument("input_json", help="Path to input JSON from extract_pdf")
    parser.add_argument("--output_dir", default="vectorstore", help="Directory to save the FAISS index")
    
    args = parser.parse_args()
    embed_data(args.input_json, args.output_dir)
