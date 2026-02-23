import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables (e.g., Hugging Face API key)
load_dotenv()

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chatbot"
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("🤖 AI Chatbot with Hugging Face + LangChain")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("ai").write(msg["content"])

# Chat input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI response
    result = model.invoke(user_input)
    response = result.content

    # Save AI response
    st.session_state["messages"].append({"role": "ai", "content": response})
    st.chat_message("ai").write(response)
