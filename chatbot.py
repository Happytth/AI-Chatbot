import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# (Optional) For local development only — won't break cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Get API key from environment (Streamlit Secrets or local .env)
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="chatbot",
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.title("🤖 AI Chatbot with Hugging Face + LangChain")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    role = "user" if msg["role"] == "user" else "assistant"
    st.chat_message(role).write(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").write(user_input)

    # AI response
    result = model.invoke(user_input)
    response = result.content

    st.session_state["messages"].append(
        {"role": "assistant", "content": response}
    )
    st.chat_message("assistant").write(response)
