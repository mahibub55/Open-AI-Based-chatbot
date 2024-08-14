import streamlit as st
from spellchecker import SpellChecker
from chat import get_chatbot_response

def main():
    st.title("Chatbot")
    message = st.text_input("Enter your message:")
    if st.button("Ask"):
        bot_response = get_bot_response(message)
        st.text(bot_response)

def get_bot_response(message):
    spell = SpellChecker()
    corrected_message = ' '.join([spell.correction(word) for word in message.split()])
    bot_response = get_chatbot_response(corrected_message).capitalize()
    return bot_response

if __name__ == "__main__":
    main()
