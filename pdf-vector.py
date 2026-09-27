from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from google import genai
import os
import faiss
import numpy as np

# Load environment variables from .env
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

pdf_path = "Data_Storage_Architecture_Tanay_P_Thakare (1).pdf"


# --------------------------------------------------
# 1. Read PDF
# --------------------------------------------------

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() or ""


# --------------------------------------------------
# 2. Split PDF text into chunks
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 3. Create embeddings for all chunks
# --------------------------------------------------

embeddings = []

for chunk in chunks:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )

    embedding = response.embeddings[0].values

    embeddings.append(embedding)


print("Embeddings created successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding size:", len(embeddings[0]))


# --------------------------------------------------
# 4. Convert embeddings to NumPy array
# --------------------------------------------------

embedding_array = np.array(
    embeddings,
    dtype="float32"
)


# --------------------------------------------------
# 5. Create FAISS index
# --------------------------------------------------

index = faiss.IndexFlatL2(
    embedding_array.shape[1]
)


# --------------------------------------------------
# 6. Add embeddings to FAISS
# --------------------------------------------------

index.add(embedding_array)

print("FAISS index created successfully!")
print("Total vectors in FAISS:", index.ntotal)


# 7. Save FAISS index
faiss.write_index(index, "faiss_index.bin")

# Save the original chunks
with open("chunks.txt", "w", encoding="utf-8") as file:
    for chunk in chunks:
        file.write(chunk.replace("\n", " ") + "\n")

print("FAISS index and chunks saved successfully!")