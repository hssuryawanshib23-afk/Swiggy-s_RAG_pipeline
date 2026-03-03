import os
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import exactly the same logic from app.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for the pipeline
history_aware_retriever = None
question_answer_chain = None
fallback_question_answer_chain = None
chat_history = []

def load_rag_pipeline():
    global history_aware_retriever, question_answer_chain, fallback_question_answer_chain
    print("Loading RAG Pipeline (This runs ONCE)...")
    
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Load Existing FAISS Index (NO RE-TRAINING/CHUNKING)
    vectorstore_path = "vectorstore/faiss_index"
    vectorstore = FAISS.load_local(
        vectorstore_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 3. Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=2000
    )

    # 3b. Initialize Gemini Fallback LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=2000,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # 4. Create History-Aware Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 5. Build QA Chain
    qa_system_prompt = """You are an expert financial AI assistant answering questions based solely on the Swiggy Annual Report FY 2023-24. 
    Use the following pieces of retrieved context to answer the user's question. 
    
    CRITICAL INSTRUCTIONS:
    - ONLY use the provided context.
    - If the answer is not contained in the context, explicitly state "I cannot find the answer in the provided documents." Do NOT guess or hallucinate.
    - If asked about financial numbers, cite the specific numbers exactly as they appear in the context.
    - Keep your answer clear and well-structured.
    
    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    fallback_question_answer_chain = create_stuff_documents_chain(gemini_llm, qa_prompt)
    print("Pipeline Loaded Successfully!")

# Load the pipeline when starting the app
load_rag_pipeline()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.json
    user_query = data.get("message")
    
    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    try:
        start_retrieval = time.time()
        sources = history_aware_retriever.invoke({"input": user_query, "chat_history": chat_history})
        retrieval_time = time.time() - start_retrieval
        
        start_llm = time.time()
        fallback_used = False
        try:
            answer = question_answer_chain.invoke({
                "input": user_query,
                "chat_history": chat_history,
                "context": sources
            })
        except Exception as e:
            print(f"Groq API failed: {e}. Falling back to Gemini...")
            fallback_used = True
            answer = fallback_question_answer_chain.invoke({
                "input": user_query,
                "chat_history": chat_history,
                "context": sources
            })
        llm_time = time.time() - start_llm
        
        # Format Sources
        sources_list = []
        if "I cannot find the answer" in answer:
            pass # No sources if no answer
        else:
            unique_pages = set()
            for doc in sources:
                page = doc.metadata.get("page", "Unknown")
                doc_type = doc.metadata.get("type", "Unknown")
                if page not in unique_pages:
                    sources_list.append({"page": page, "type": doc_type})
                    unique_pages.add(page)
            
        # Update chat history
        chat_history.append(HumanMessage(content=user_query))
        clean_content = answer.split("\n\n**Sources:**")[0] # Just in case the LLM appends it
        
        if fallback_used:
            clean_content += "\n\n*(Note: Groq API limit reached or unavailable. This response was generated using the Gemini 2.5 Flash fallback model.)*"
            
        chat_history.append(AIMessage(content=clean_content))
        
        # Optional: Limit history size to prevent context overflow
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({
            "response": clean_content,
            "sources": sources_list,
            "retrieval_time": retrieval_time,
            "generation_time": llm_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
