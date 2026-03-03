import re
import os
import time

import markdown as md_lib

def render_to_html(text):
    """Convert LLM output to fully styled HTML server-side."""
    # Force blank line before every **Heading**: pattern
    text = re.sub(r'(?<!\n)(\*\*[A-Z][^*\n]+\*\*\s*:)', r'\n\n\1', text)
    # Bold percentages, ₹ amounts, share counts
    text = re.sub(r'(\b\d+\.?\d*%)', r'**\1**', text)
    text = re.sub(r'(₹\s*[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion))?)', r'**\1**', text, flags=re.IGNORECASE)
    text = re.sub(r'(\([\d,]+ shares?\))', r'**\1**', text)
    # KEY FIX: every single newline becomes a double newline
    # so each sentence/line becomes its own <p> block with margin
    text = re.sub(r'\n{3,}', '\n\n', text)   # collapse 3+ to 2 first
    text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text)  # single \n → \n\n
    # Convert markdown → HTML (no nl2br needed now)
    html = md_lib.markdown(text.strip())
    # Inject inline styles
    html = html.replace('<p>', '<p style="margin:0.8rem 0;line-height:1.75;">')
    html = html.replace('<li>', '<li style="margin:0.55rem 0;line-height:1.75;">')
    html = html.replace('<ul>', '<ul style="padding-left:1.4rem;margin:0.6rem 0;list-style-type:disc;">')
    html = html.replace('<ol>', '<ol style="padding-left:1.4rem;margin:0.6rem 0;list-style-type:decimal;">')
    html = html.replace('<h2>', '<h2 style="font-size:1.1rem;font-weight:700;margin:1.2rem 0 0.4rem;">')
    html = html.replace('<h3>', '<h3 style="font-size:1rem;font-weight:700;margin:1rem 0 0.35rem;">')
    html = html.replace('<strong>', '<strong style="font-weight:700;">')
    return html

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

# Global variables for the pipeline
history_aware_retriever = None
question_answer_chain = None
fallback_question_answer_chain = None
fallback_history_aware_retriever = None
chat_history = []

# Rate-limit tracker: if Groq returns 429, record when it becomes available again
groq_throttled_until = 0  # Unix timestamp; 0 means not throttled
gemini_throttled_until = 0

def use_groq() -> bool:
    """Return True only if Groq is not currently rate-limited."""
    return time.time() >= groq_throttled_until

def use_gemini() -> bool:
    return time.time() >= gemini_throttled_until

def mark_groq_throttled(error_msg: str):
    """Parse retry-after seconds from the 429 error message and block Groq until then."""
    global groq_throttled_until
    match = re.search(r'(\d+)m([\d.]+)s', str(error_msg))
    if match:
        minutes, seconds = int(match.group(1)), float(match.group(2))
        wait = minutes * 60 + seconds + 5
    else:
        match2 = re.search(r'try again in ([\d.]+)s', str(error_msg))
        wait = float(match2.group(1)) + 5 if match2 else 900
    groq_throttled_until = time.time() + wait
    print(f"Groq throttled for {wait:.0f}s (until {time.strftime('%H:%M:%S', time.localtime(groq_throttled_until))})")

def mark_gemini_throttled(error_msg: str):
    global gemini_throttled_until
    match = re.search(r'retry in ([\d.]+)s', str(error_msg), re.IGNORECASE)
    if match:
        wait = float(match.group(1)) + 2
    else:
        match2 = re.search(r'(\d+)m([\d.]+)s', str(error_msg))
        if match2:
            wait = int(match2.group(1)) * 60 + float(match2.group(2)) + 5
        else:
            wait = 65  # default 65s (per-minute quota)
    gemini_throttled_until = time.time() + wait
    print(f"Gemini throttled for {wait:.0f}s (until {time.strftime('%H:%M:%S', time.localtime(gemini_throttled_until))})") 

def load_rag_pipeline():
    global history_aware_retriever, question_answer_chain, fallback_question_answer_chain, fallback_history_aware_retriever
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
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Initialize Groq LLMs
    #    - Small model for question rephrasing (saves tokens)
    #    - Big model for answering (quality)
    llm_rephrase = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=256
    )
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=1024
    )

    # 3b. Initialize Gemini Fallback LLM
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.0,
        max_tokens=1024,
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
        llm_rephrase, retriever, contextualize_q_prompt
    )

    # 5. Build QA Chain
    qa_system_prompt = """You are an expert financial AI assistant. You MUST format ALL responses using Markdown syntax — this is mandatory, not optional.

    FORMATTING RULES (always apply these):
    - Use ## for section headings
    - Use **bold** for all company names, numbers, and percentages
    - Use bullet points (- ) for any list of items
    - Add a blank line between sections

    CONTENT RULES:
    - Answer based solely on the Swiggy Annual Report FY 2023-24.
    - ONLY use the provided context. If the answer is not in the context, say: "I cannot find the answer in the provided documents."
    - Cite financial numbers exactly as they appear in the context.

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
    fallback_history_aware_retriever = create_history_aware_retriever(
        gemini_llm, retriever, contextualize_q_prompt
    )
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
        start_time = time.time()
        fallback_used = False
        answer = None
        retrieval_time = 0
        llm_time = 0

        def call_gemini(query, history, ctx):
            """Call Gemini retriever + chain, raise on failure."""
            src = fallback_history_aware_retriever.invoke({"input": query, "chat_history": history})
            ans = fallback_question_answer_chain.invoke({"input": query, "chat_history": history, "context": ctx if ctx else src})
            return src, ans

        if not use_groq():
            print("Groq still throttled — using Gemini directly.")
            fallback_used = True
            if not use_gemini():
                sources, answer = [], "Both APIs are rate-limited. Please try again in a minute."
            else:
                try:
                    start_ret = time.time()
                    sources, answer = call_gemini(user_query, chat_history, None)
                    total = time.time() - start_ret
                    retrieval_time = total * 0.3
                    llm_time = total * 0.7
                except Exception as gem_err:
                    gem_str = str(gem_err)
                    if '429' in gem_str or 'quota' in gem_str.lower():
                        mark_gemini_throttled(gem_str)
                    print(f"Gemini also failed: {gem_err}")
                    sources, answer = [], "Both APIs are rate-limited. Please try again in a minute."
        else:
            try:
                start_ret = time.time()
                sources = history_aware_retriever.invoke({"input": user_query, "chat_history": chat_history})
                retrieval_time = time.time() - start_ret
                start_llm = time.time()
                answer = question_answer_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history,
                    "context": sources
                })
                llm_time = time.time() - start_llm
            except Exception as groq_err:
                error_str = str(groq_err)
                if '429' in error_str or 'rate_limit' in error_str.lower() or 'tpd' in error_str.lower():
                    mark_groq_throttled(error_str)
                print(f"Groq failed: {groq_err}. Falling back to Gemini...")
                fallback_used = True
                if not use_gemini():
                    sources, answer = [], "Both APIs are rate-limited. Please try again in a minute."
                else:
                    try:
                        start_ret = time.time()
                        sources, answer = call_gemini(user_query, chat_history, None)
                        total = time.time() - start_ret
                        retrieval_time = total * 0.3
                        llm_time = total * 0.7
                    except Exception as gem_err:
                        gem_str = str(gem_err)
                        if '429' in gem_str or 'quota' in gem_str.lower():
                            mark_gemini_throttled(gem_str)
                        print(f"Gemini also failed: {gem_err}")
                        sources, answer = [], "Both APIs are rate-limited. Please try again in a minute."

        # Convert to styled HTML server-side
        answer_html = render_to_html(answer)
        
        # Format Sources — always show retrieved pages (even if LLM hedges)
        sources_list = []
        print(f"[DEBUG-SOURCES] type(sources)={type(sources).__name__}, len={len(sources) if isinstance(sources, list) else 'N/A'}")
        if isinstance(sources, list) and len(sources) > 0:
            print(f"[DEBUG-SOURCES] First doc type={type(sources[0]).__name__}, metadata={getattr(sources[0], 'metadata', 'NO_METADATA')}")
        unique_pages = set()
        for doc in sources:
            page = doc.metadata.get("page", "Unknown")
            doc_type = doc.metadata.get("type", "Unknown")
            if page not in unique_pages:
                sources_list.append({"page": page, "type": doc_type})
                unique_pages.add(page)
        print(f"[DEBUG-SOURCES] Final sources_list = {sources_list}")
            
        # Update chat history (store plain answer text, not HTML)
        chat_history.append(HumanMessage(content=user_query))
        clean_content = answer.split("\n\n**Sources:**")[0]
        if fallback_used:
            answer_html += '<p style="margin:0.8rem 0;font-style:italic;opacity:0.7;">(Note: Groq API limit reached. Response generated using Gemini fallback.)</p>'
        chat_history.append(AIMessage(content=clean_content))
        
        # Optional: Limit history size to prevent context overflow
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        return jsonify({
            "response": answer_html,
            "sources": sources_list,
            "retrieval_time": retrieval_time,
            "generation_time": llm_time
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return as a chat message, not an HTTP error, so the UI shows it cleanly
        return jsonify({
            "response": render_to_html("Sorry, an unexpected error occurred. Please try again."),
            "sources": [],
            "retrieval_time": 0,
            "generation_time": 0
        })

@app.route("/api/clear_chat", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)