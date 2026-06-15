# 🎓 Student Success Analytics Platform

An enterprise-grade, hybrid RAG-based analytics ecosystem designed to transform static academic documents (Syllabuses, PYQs, and Notes) into interactive, hyper-speed exam preparation insights. Built with modern microservice architecture, this platform empowers students to dynamically generate scannable revision notes, formulate subject-aware tactical study schedules, and access personalized career paths natively.

---

## 🎯 Project Overview & Core Mission

The **Student Success Analytics Platform** is engineered to eliminate study fragmentation. By digesting disparate academic materials, our Retrieval-Augmented Generation (RAG) engine instantly cross-references course contexts to provide a singular pane of glass for all academic preparation. From automated syllabus coverage matrices to hyper-customized timeline planning, this system is mission-critical for optimizing academic success trajectories.

## 🏗️ Decoupled System Architecture

The ecosystem leverages a decoupled microservice layout for maximum scalability, fault isolation, and independent deployment cycles.

- **Frontend Client:** A Streamlit Cloud application providing highly responsive, state-driven dynamic workspace interfaces. It natively manages user session controls, instantaneous visual feedback widgets, and highly optimized sequential asset streaming.
- **Core Compute Backend:** A robust FastAPI engine deployed securely within a Render production container. This handles heavy-duty advanced text extraction pipelines, structural chunk transformation parsing, and automated dense vector mappings.

## 🛠️ The Tech Stack Grid

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend UI** | Streamlit Client | Dynamic component state & reactive interfaces |
| **API Framework** | FastAPI (Uvicorn ASGI) | High-performance, asynchronous REST architecture |
| **Vector Storage** | Pinecone DB | Low-latency dense vector spatial indexing |
| **Metadata Caching** | MongoDB Atlas Cloud | Scalable user schemas & history persistence |
| **Language Generation** | Groq Cloud API | High-speed LLM text generation (`llama-3.3-70b-versatile`) |
| **Document Embedding** | Google AI Studio SDK | Deep vector spatial mappings (`gemini-embedding-001`) |

## 🚀 Advanced Production Optimizations (Our Key Engineering Wins!)

To ensure enterprise-level uptime and seamless user flow under high traffic, we hardened the entire environment against severe cloud infrastructure bottlenecks:

- **Cloud Ingress Size Limitation Patch (HTTP 413 Resolution):** 
  Transformed a volatile concurrent file bundling pipeline into a strict, memory-efficient sequential frontend transmission loop. By enforcing explicit `gc.collect()` triggers in the backend parsing layers, the engine remains safely and predictably within Render's free 512MB RAM constraints, successfully dodging OOM (Out-of-Memory) resets during bulk ingestion.

- **Transient Server Exception Mitigation (HTTP 503 Backoff Shielding):** 
  Integrated automated, exponential backoff timing parameters into the ETL layer utilizing the `tenacity` retry framework. This insulates our vector generation pipelines entirely from third-party AI server drops, seamlessly absorbing cloud anomalies without triggering hard crashes.

- **Dynamic Multi-Account Fallbacks:** 
  Engineered resilient parsing checks to guarantee smooth runtime processing when handling backwards compatibility. The system safely maneuvers around structural schema mismatches in older user arrays and silently absorbs localized API quota exhaustion events.

## 🔑 Required Environment Configurations (.env template)

To run this platform locally or spin up a new cloud instance, the following secure keys must be provisioned.

```env
# --- AI Providers ---
GEMINI_API_KEY="AIzaSy..."
GROQ_API_KEY="gsk_..."

# --- Vector Database ---
PINECONE_API_KEY="xxxxxx-xxxx-xxxx..."
PINECONE_INDEX_NAME="pdf-rag-etl"

# --- Primary Database & Auth ---
MONGODB_URI="mongodb+srv://<user>:<pass>@cluster..."
JWT_SECRET="super-secret-production-hash-key"

# --- Frontend Connection String ---
# BACKEND_URL="http://backend:8000/api/v1" # Local docker network fallback
```

> [!WARNING]  
> **Security Notice:** Never commit your production `.env` file to version control. Always verify that `.env` is explicitly listed within your `.gitignore` tracking before pushing changes.
