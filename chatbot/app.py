from flask import Flask, render_template, request, jsonify
from spellchecker import SpellChecker
from chat import get_chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText']

    # Spelling correction
    spell = SpellChecker()
    corrected_message = ' '.join([spell.correction(word) for word in message.split()])

    # Get response from the chatbot
    bot_response = get_chatbot_response(corrected_message).capitalize()

    return jsonify({'status': 'OK', 'answer': bot_response})

if __name__ == "__main__":
    app.run(debug=True)
