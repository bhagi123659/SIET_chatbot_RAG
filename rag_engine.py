python
import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Load PDFs
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    mypages = [i.extract_text() or "" for i in reader.pages]
    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable text found")

    return full_text


# Chunking
def create_chunks(full_text, chunksize=500, overlap=100):
    if chunksize <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunksize:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# Embedding
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = embed_vectors.astype("float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# Call load_pdf, chunk, embedding
def load_base(pdf_path="01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)

    chunks = create_chunks(full_text)

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# Retrieve
def retrive_content(question, embed_model, chunks, index, num_chunks=3):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one or more chunks")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = query_embed.astype("float32")

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# Ask
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("API key missing")

    # Retrieve relevant content
    background = retrive_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks=num_chunks
    )

    # Create prompt
    prompt = f"""
    You are a professional AI assistant.

    Answer the user's question using only the document background below.

    Do not invent or assume facts.

    If information is not available, simply say:
    Information missing from the document.

    Document Background:
    {background}

    User Question:
    {question}

    Answer:
    """

    # Connect to Groq
    client = Groq(api_key=api_key)

    # Generate response
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# Main execution
if __name__ == "__main__":

    index, embed_model, chunks = load_base(
        "01_College_Information.pdf"
    )

    api_key = "YOUR_GROQ_API_KEY"

    question = "What courses are available in the college?"

    answer = ask_rag(
        question,
        embed_model,
        chunks,
        index,
        api_key
    )

    print("AI Answer:", answer
