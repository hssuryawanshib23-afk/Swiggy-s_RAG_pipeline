import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import textwrap

# Load environment variables
load_dotenv()

def init_rag_pipeline(vectorstore_dir="vectorstore"):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    print("Loading FAISS vector store...")
    # Allow dangerous deserialization since we created this local FAISS index ourselves
    vectorstore = FAISS.load_local(
        os.path.join(vectorstore_dir, "faiss_index"), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Configure retrieving: return top 5 chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print("Initializing Groq LLM (Llama 3.3 70B Versatile)...")
    # Llama 3.3 70B Versatile is currently one of the best available on Groq for RAG
    # It has a high context window, great reasoning, and responds instantly.
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0, # 0.0 for strict factuality in RAG
        max_tokens=1024
    )

    # Prompt Template for Financial RAG
    # This enforces strict grounding to avoid hallucinated numbers
    template = """
    You are an AI assistant designed to answer questions based on the Swiggy Annual Report. 
    Use the following pieces of retrieved context to answer the user's question. 
    
    CRITICAL INSTRUCTIONS:
    - If the answer is not contained in the context, explicitly state "I cannot find the answer in the provided documents." Do NOT guess or hallucinate.
    - If asked about financial numbers, cite the specific numbers exactly as they appear in the context.
    - If the context contains tabular data, try to interpret the table accurately.
    
    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Build RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

def run_chat_loop():
    print("\n" + "="*50)
    print("🚀 Swiggy RAG Pipeline is ready!")
    print("Powered by Groq Llama 3.3 70B & FAISS")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    qa_chain = init_rag_pipeline()
    
    while True:
        query = input("\n📝 Ask a question about the Swiggy Annual Report: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("\n⏳ Thinking...")
        try:
            result = qa_chain.invoke({"query": query})
            
            # Print the Answer
            print("\n" + "="*50)
            print("🤖 Answer:")
            print(textwrap.fill(result["result"], width=80))
            print("="*50)
            
            # Print Sources for transparency
            print("\n📚 Sources Used:")
            sources = result.get("source_documents", [])
            for i, doc in enumerate(sources):
                page = doc.metadata.get("page", "Unknown")
                doc_type = doc.metadata.get("type", "Unknown")
                print(f"  - Source {i+1}: Page {page} (Type: {doc_type})")
                
        except Exception as e:
            print(f"\n❌ Error during retrieval/generation: {e}")

if __name__ == "__main__":
    run_chat_loop()
