import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    # Safe JSON parsing (Gemini ရဲ့ Feedback အရ ပြင်ထားတယ်)
    data = request.get_json(silent=True) or {}
    
    mode = data.get('mode', 'General')
    message = data.get('message', '')
    debate_on = data.get('debateOn', False)

    # Phase 2: Mock Response (AI အစစ်ကို Phase 3 မှာထည့်မယ်)
    response_text = f"📡 Backend Connected!\nMode: {mode}\nDebate: {'ON' if debate_on else 'OFF'}\n\nYou said: {message}"

    return jsonify({
        "response": response_text,
        "mode": mode,
        "debateOn": debate_on
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
