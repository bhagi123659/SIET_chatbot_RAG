import streamlit as st

from rag_engine import load_base, ask_rag


# Page configuration
st.set_page_config(
    page_title="College Information Chatbot",
    page_icon="🎓"
)


# Application title
st.title("🎓 College Information Chatbot")

st.write(
    "Ask questions about college information "
    "using the PDF knowledge base."
)


# Load PDF and build vector database
@st.cache_resource
def initialize_rag():

    index, embed_model, chunks = load_base(
        "01_College_Information.pdf"
    )

    return index, embed_model, chunks


# Initialize RAG
try:

    index, embed_model, chunks = initialize_rag()

    st.success("College knowledge base loaded successfully!")

except Exception as e:

    st.error(f"PDF loading error: {str(e)}")

    st.stop()


# Get API key
try:

    my_key = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "Groq API key not found. "
        "Please configure GROQ_API_KEY in Streamlit Secrets."
    )

    st.stop()


# User input
question = st.text_input(
    "Enter your question:"
)


# Submit button
if st.button("Ask Question"):

    if question.strip():

        try:

            with st.spinner("Generating answer..."):

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=my_key,
                    num_chunks=3
                )

            st.subheader("AI Answer")

            st.write(answer)

        except Exception as e:

            st.error(f"Groq API Error: {str(e)}")

    else:

        st.warning("Please enter a question.")
