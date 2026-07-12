# AI Lecture Assistant - RAG Chatbot

Ask questions about AI/ML lectures and instantly get accurate answers with the exact video and timestamp where the topic is taught.

⸻

## Overview

Learning from long lecture videos can be time-consuming. Finding where a specific concept is explained often requires manually searching through hours of content.

AI Lecture Assistant solves this problem using Retrieval-Augmented Generation (RAG).

Users can ask questions in natural language, and the system:

* 🔍 Searches the lecture transcripts semantically.
* 🧠 Retrieves the most relevant lecture segments.
* 🤖 Generates answers using an LLM.
* 🎥 Recommends the exact video and timestamp where the concept is taught.

⸻

## Features

* 🎤 Speech-to-Text using OpenAI Whisper
* ✂️ Automatic transcript chunking
* 🧠 Semantic embeddings using nomic-embed-text
* ⚡ Fast vector search with FAISS
* 🤖 AI-powered answers using Gemini
* 🎯 Video title & timestamp recommendations
* 📚 Context-aware Retrieval-Augmented Generation (RAG)
* 💻 Fully local retrieval pipeline

⸻

## System Architecture

                YouTube Videos
                        │
                        ▼
                OpenAI Whisper
                        │
                        ▼
                 Video Transcripts
                        │
                        ▼
                    Chunking
                        │
                        ▼
             nomic-embed-text
                  Embeddings
                        │
                        ▼
                 FAISS Index
                        │
                        ▼
                 User Question
                        │
                        ▼
              Question Embedding
                        │
                        ▼
             Semantic Similarity
                        │
                        ▼
              Top Relevant Chunks
                        │
                        ▼
                Prompt Generation
                        │
                        ▼
                 Gemini LLM
                        │
                        ▼
         Answer + Video + Timestamp

⸻

## Tech Stack

Component	Technology
Language	Python
Speech-to-Text	OpenAI Whisper
Embedding Model	nomic-embed-text (Ollama)
Vector Search	FAISS
LLM	Gemini
Data Processing	Pandas, NumPy
Similarity Search	FAISS
IDE	VS Code

⸻

## Project Structure

RAG_Project/
│
├── audios/
├── videos/
├── jsons/
├── data/
│   ├── embeddings.pkl
│   └── vector_index.faiss
│
├── main.py
├── embedding.py
├── prompt.py
├── llm_response.py
├── cosine_similar_vectors.py
├── faiss_utils.py
├── requirements.txt
├── README.md
└── .gitignore

⸻

## How It Works

1. Download Lecture Videos

Download AI/ML lecture videos.

⬇️

2. Speech-to-Text

Convert lecture audio into text using Whisper.

⬇️

3. Chunking

Split transcripts into manageable chunks.

⬇️

4. Generate Embeddings

Convert each chunk into vector embeddings using nomic-embed-text.

⬇️

5. Build FAISS Index

Store embeddings for fast semantic retrieval.

⬇️

6. User Query

Convert the user’s question into an embedding.

⬇️

7. Semantic Retrieval

Retrieve the most relevant lecture chunks.

⬇️

8. Prompt Generation

Build a context-aware prompt using retrieved chunks.

⬇️

9. Gemini Response

Generate an accurate answer based only on retrieved content.

⬇️

10. Learning Recommendation

Return:

* ✅ Answer
* 🎥 Video Title
* ⏱️ Timestamp

⸻

 Example

User Question

What is a Multi Layer Perceptron?

Response

A Multi Layer Perceptron (MLP) is a neural network consisting of multiple layers of neurons. It can learn complex, non-linear relationships and is trained using backpropagation.
Video:
Multi Layer Perceptron | MLP Intuition
Timestamp:
00:05:12 – 00:09:46

⸻

## Installation

git clone <repository-url>
cd RAG_Project
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

⸻

## Environment Variables

Create a .env file:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

⸻

## Run

python main.py

⸻

## Future Improvements

* Streamlit Web UI
* Chat History
* ChromaDB Integration
* Hybrid Search (BM25 + Vector Search)
* Cross-Encoder Re-ranking
* Clickable YouTube Timestamp Links
* Multi-Playlist Support
* PDF & PPT Support
* Voice Input & Voice Output
* Docker Deployment

⸻

## Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* FAISS
* Whisper
* Prompt Engineering
* LLM Integration
* Information Retrieval
* AI Application Development

⸻

🤝 Contributing

Contributions, suggestions, and improvements are always welcome.

Feel free to fork the repository and submit a pull request.

⸻

⭐ Support

If you found this project useful, consider giving it a ⭐ Star on GitHub.

It helps others discover the project and motivates future improvements.
