from twilio.rest import Client # type: ignore
import time
from pathlib import Path
# THis code is working fine for initial stage
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = {"reply": f"You said: {user_input}"}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
