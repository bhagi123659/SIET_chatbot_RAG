import stream as st
from rag_engine import load_base,ask_rag
st.set_page_config(page_titel = "SIET_chatbot",layout = "centered")
st.titel("shridevi Institute of Engineering and Techology - chatbot")
st.caption("ask anything about college information...")
st.caption("created by bhagirati s")

myKey = st.secrects("GROQ_API_KEY")

@st.cache_resource
def intiate_rag():
  return load_base("01_college_Information.pdf")

index,embed_model,chunks = initiate_rag()

if "chat_hist" not in st.session_state:
  st.session_state.chat_hist = []

for qa in st.session_state.chat_hist :
  with st.chat_message(qa["role"]):
    st.write(qa["content"])

question = st.chat_input("Ask anything about SIET college...")

if question:
  st.session_state.chat_hist.append(
    {"role" :"user", content = qestion})
  with st.chat_message("user"):
    st.write(question)
    
  with st.chat_message("assistant"):
    with st.spinner("Asking RAG for answer..."):
      answer = ask_rag(question, embed_model, chunks, index, api_key = mykey, num_chunks)
      st.write(answer)
      
if st.button("clear chat")
st.session_state.chat_hist = []
st.rerun()
