from flask import Flask, render_template # type: ignore
from flask_socketio import SocketIO, emit # type: ignore

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e0f08f6b9c5b3f2d1a7c3e74a9d6b8a5'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def home():
    return "Chatbot WebSocket Server is running!"

@socketio.on('message')
def handle_message(data):
    print(f"Received message: {data}")
    user_input = data['message']
    # Generate chatbot response
    response = {"reply": f"Chatbot response to: {user_input}"}
    emit('response', response)

if __name__ == '__main__':
    socketio.run(app, debug=True)
