import streamlit as st
from rag_engine import load_base, ask_rag


# Page configuration
st.set_page_config(
    page_title="SIET Chatbot",
    layout="centered"
)


# Application heading
st.title(
    "Shridevi Institute of Engineering and Technology - ChatBot"
)

st.caption("Ask anything about college information...")

st.write("Created by Aishwarya S M")


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

except Exception as e:

    st.error(f"PDF loading error: {str(e)}")
    st.stop()


# Get Groq API key
try:

    my_key = st.secrets["GROQ_API_KEY"]

    if not my_key or not my_key.strip():
        st.error("Groq API key is empty.")
        st.stop()

except KeyError:

    st.error(
        "Groq API key not found. "
        "Please configure GROQ_API_KEY in Streamlit Secrets."
    )

    st.stop()


# Initialize chat history
if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous chat messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# Clear chat button
if st.button("Clear chat"):

    st.session_state.messages = []

    st.rerun()


# Chat input
question = st.chat_input(
    "Ask anything about SIET college..."
)


# Process user question
if question:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.write(question)


    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            try:

                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=my_key,
                    num_chunks=3
                )

                st.write(answer)

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Error: {str(e)}")
