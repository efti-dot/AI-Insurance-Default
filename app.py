import streamlit as st
from main import process_pdf, process_attachment, query
from prompt import OpenAIConfig
from dotenv import load_dotenv
import os

api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("Please check the OPENAI_API_KEY.")

ai = OpenAIConfig(api_key=api_key)

def AI_insurance_assistance():
    st.title("AI Insurance Assistance")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask anything...")
    attached_file = st.file_uploader("Attach a file to this message", type=["pdf","docx","pptx","ppt","png","jpg","jpeg"])

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        if attached_file:
            msg = process_attachment(attached_file)
            st.success(msg)

        response = ai.get_response(user_input, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Reset Session"):
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    AI_insurance_assistance()

