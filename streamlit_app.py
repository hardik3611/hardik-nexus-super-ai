import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 1. Set up the title of your web app
st.title("Nexus Chat Bot")

# 2. Cache the model loading so your app stays fast
@st.cache_resource
def load_chatbot():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chat_bot = load_chatbot()

# 3. Create the chat history structure
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. Handle new user inputs
if user_query := st.chat_input("Ask Nexus anything..."):
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # Generate response from the Qwen AI brain
    with st.chat_message("assistant"):
        with st.spinner("Nexus is thinking..."):
            conversation = [{"role": "user", "content": user_query}]
            output = chat_bot(conversation, max_new_tokens=250)
            response_text = output['generated_text'][-1]['content']
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
