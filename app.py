import streamlit as st
import json


# Function to load the response from the saved file
def load_response_from_file(filename="response.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


st.title("TW@Porsche Chatbot")
st.write("This is a simple chatbot demo powered by our fine-tuned LLM model.")

# Initialize chat history if it doesn't exist in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load the saved response from the JSON file
    response_data = load_response_from_file()  # Load the saved response

    # Extract the assistant's response
    assistant_response = response_data.get("assistant_response", "Sorry, I couldn't find a response.")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
