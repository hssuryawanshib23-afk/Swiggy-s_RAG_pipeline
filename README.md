---
title: Swiggy Elite AI
emoji: 🍕
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Swiggy Elite AI: RAG Pipeline on Annual Report

This project is a complete Retrieval-Augmented Generation (RAG) Artificial Intelligence application built to confidently answer complex financial questions based entirely on the **Swiggy Annual Report FY 2023-24**. It strictly grounds responses in the provided financial document to eliminate hallucinations.

**Demo Source Document**: [Swiggy Annual Report](https://www.swiggy.com/) (Ensure this matches your actual hosted or downloaded PDF source)

---

## 🛠 Project Architecture & Approach

As outlined in the ML Intern Assignment requirements, the application tackles the challenge efficiently:

### 1. Document Processing (`extract_pdf.py` & `embed_data.py`)

- We loaded the raw Swiggy Annual Report PDF.
- Text was pre-processed using OCR fallback for images and tabular data structure retention (`pdfplumber`).
- Meaningful and cohesive chunks were generated using LangChain’s `RecursiveCharacterTextSplitter`.

### 2. Embedding & Vector Store

- **Embeddings:** We utilize the BAAI `bge-small-en-v1.5` embeddings (an efficient, highly-ranked open-source HuggingFace model) to convert the chunks into dense vector representations.
- **Vector Database:** These vectors are then indexed into a persistent **FAISS** (Facebook AI Similarity Search) database locally, optimizing semantic similarity search without requiring cloud vector databases.

### 3. Retrieval-Augmented Generation (RAG) (`main.py`)

- The backend application utilizes a **History-Aware Retriever Chain**. This analyzes the user query _and_ chat history context, retrieves the Top-K (`k=10`) most matching FAISS chunks, and funnels this exact context into the LLM.
- **LLM Engine:** Powered primarily by the lightning-fast **Groq API** running `llama-3.3-70b-versatile`.
- **Graceful Fallback:** Implemented a robust `try/catch` failover—if Groq limits are hit, the pipeline instantly reroutes the RAG context to **Gemini 2.5 Flash** (`langchain-google-genai`).

### 4. Question Answering Interface (UI)

- A highly polished, robust **Flask** web application combined with a responsive **Tailwind CSS** frontend (`index.html`).
- Features a dark/light mode toggle, dynamic categorization of pre-set prompts natively inside the UI, millisecond performance metrics, and strict source grounding.

---

## 🚀 How to Run Locally

1. **Clone the Repository** and navigate to the project directory:

   ```bash
   git clone <your-repo-link>
   cd Swiggy_RAG_pipeline
   ```

2. **Set up the Virtual Environment & Dependencies**
   (Ensure you have Python 3.9+ installed)

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   Create a `.env` file in the root directory (do not commit this file to GitHub!):

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Boot the FAISS Vector Database & Server**

   ```bash
   python main.py
   ```

   _The script will load the BAAI embeddings and preexisting FAISS store into memory (~15 seconds)._

5. **Interact**
   Open your browser to `http://127.0.0.1:5000/`.

---

## ☁️ How to Host on the Cloud (Render / Railway)

Because the project loads HuggingFace embedding models into RAM along with the FAISS vector indices, it generally requires **~1GB to 2GB** of server RAM to boot smoothly without crashing. Free tiers on most PaaS platforms may fail with "Out of Memory" (OOM) errors.

**Recommended Deployment via Render.com:**

1. Push your code to a **private GitHub repository** (do NOT include `.env` or `venv`).
2. Go to **Render.com** and sign in.
3. Click **New +** > **Web Service**.
4. Link your GitHub repository.
5. Apply the following backend settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn main:app`
6. Select the **Starter Tier ($7/mo)** or higher. (A free 512MB RAM tier will likely crash upon loading).
7. Scroll down to **Environment Variables** and securely add:
   - `GROQ_API_KEY` = `your_actual_key`
   - `GEMINI_API_KEY` = `your_actual_key`
8. Click **Deploy Web Service**! Once complete, you will get a permanent public URL.
