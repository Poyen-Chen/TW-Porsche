import streamlit as st
import json

# Function to load the response from the saved file
def load_response_from_file(filename="response.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data["response"]

# Title and Description
st.title("TW@Porsche Chatbot")
st.write("This is a simple chatbot powered by our fine-tuned LLM model.")

# Session State to Store Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize an empty list for chat history

# Input Section
user_input = st.text_input("You:", placeholder="Ask your question here...")

if user_input:
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Load the saved response from the JSON file
    with st.spinner("Thinking..."):
        response = load_response_from_file()  # Load the saved response

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display Chat History
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")
