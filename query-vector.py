import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
import os

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load chunks
with open("chunks.txt", "r", encoding="utf-8") as file:
    chunks = [line.strip() for line in file.readlines()]

print("RAG system loaded successfully!")
print("Total vectors:", index.ntotal)


# --------------------------------------------------
# 1. Ask the user a question
# --------------------------------------------------

question = input("\nAsk a question about the PDF: ")


# --------------------------------------------------
# 2. Create embedding for the question
# --------------------------------------------------

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_embedding = np.array(
    [response.embeddings[0].values],
    dtype="float32"
)


# --------------------------------------------------
# 3. Search FAISS for relevant chunks
# --------------------------------------------------

distances, indices = index.search(
    question_embedding,
    k=3
)


# --------------------------------------------------
# 4. Retrieve the relevant chunks
# --------------------------------------------------

retrieved_chunks = []

for i in indices[0]:
    retrieved_chunks.append(chunks[i])


# --------------------------------------------------
# 5. Display retrieved information
# --------------------------------------------------

print("\nRelevant information found:")

for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)


# --------------------------------------------------
# 6. Combine retrieved chunks into context
# --------------------------------------------------

context = "\n\n".join(retrieved_chunks)


# --------------------------------------------------
# 7. Create prompt for Gemini
# --------------------------------------------------

prompt = f"""
Answer the user's question using only the information
provided in the context below.

If the answer cannot be found in the context,
say: "I could not find the answer in the PDF."

Context:
{context}

User question:
{question}
"""


# --------------------------------------------------
# 8. Generate final answer using Gemini
# --------------------------------------------------

answer_response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=prompt
)


# --------------------------------------------------
# 9. Display final answer
# --------------------------------------------------

print("\nFinal Answer:")
print(answer_response.text)