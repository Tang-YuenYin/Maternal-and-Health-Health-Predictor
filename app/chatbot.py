import streamlit as st
import ollama

st.title("💬 llama2 (7B) Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑‍💻 User:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg['content']}")

## Generator for Streaming Tokens
def generate_response():
    response = ollama.chat(model='llama2', stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

if prompt := st.text_input("You:", key="user_input"):
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"**🧑‍💻 User:** {prompt}")
        
        st.session_state["full_message"] = ""
        response_placeholder = st.empty()
        
        for token in generate_response():
            response_placeholder.markdown(f"**🤖 Assistant:** {st.session_state['full_message']}")
        
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
        
        # Clear the text input after sending the message
        st.session_state.user_input = ""

    


    