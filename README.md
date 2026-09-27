# PDF RAG Assistant using Gemini and FAISS

A terminal-based Retrieval-Augmented Generation (RAG) application that allows users to ask questions about information contained in a PDF document.

The project extracts text from a PDF, splits the text into smaller chunks, converts the chunks into embeddings using Google's Gemini embedding model, stores the embeddings in FAISS, retrieves relevant chunks for a user question, and uses Gemini to generate a context-based answer.

## Features

- Extract text from PDF documents
- Split documents into smaller text chunks
- Generate text embeddings using Gemini
- Store and search embeddings using FAISS
- Retrieve the most relevant document chunks
- Generate answers using Gemini
- Terminal-based question answering
- API key stored securely using `.env`

## RAG Architecture

```text
PDF Document
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Gemini Embeddings
     |
     v
FAISS Vector Index
     |
     v
User Question
     |
     v
Question Embedding
     |
     v
FAISS Similarity Search
     |
     v
Relevant Chunks
     |
     v
Gemini LLM
     |
     v
Final Answer