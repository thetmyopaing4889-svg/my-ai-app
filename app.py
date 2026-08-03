import os
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from openai import OpenAI
import google.genai as genai
import requests
import io

app = Flask(__name__)
CORS(app)

# ============================================
# 1. Configuration & Built-in Providers
# ============================================
BUILTIN_PROVIDERS = {
    "deepseek":   {"base_url": "https://api.deepseek.com",           "model": "deepseek-chat"},
    "github":     {"base_url": "https://models.github.ai/inference", "model": "gpt-4o-mini"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",       "model": "deepseek/deepseek-r1:free"},
}
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# ============================================
# 2. File-based Session Storage
# ============================================
SESSION_FILE = 'sessions_data.json'

def load_sessions():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(sessions):
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving sessions: {e}")

session_store = load_sessions()

# ============================================
# 3. Helper Functions
# ============================================
def get_openai_client(provider, api_key):
    config = BUILTIN_PROVIDERS.get(provider)
    if config:
        return OpenAI(api_key=api_key, base_url=config["base_url"])
    return None

def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)

def call_ai(prompt, provider, api_key, custom_model=None):
    if not api_key: return f"Error: No API key provided for {provider}."
    try:
        if provider == "gemini":
            client = get_gemini_client(api_key)
            model_name = custom_model or GEMINI_DEFAULT_MODEL
            resp = client.models.generate_content(model=model_name, contents=prompt)
            return resp.text.strip()
        else:
            client = get_openai_client(provider, api_key)
            config = BUILTIN_PROVIDERS.get(provider)
            if not client or not config: return f"Error: {provider} configuration issue."
            model_name = custom_model or config["model"]
            resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], timeout=30)
            return resp.choices[0].message.content
    except Exception as e:
        return f"Error ({provider}): {str(e)}"

def call_custom_ai(base_url, api_key, model_name, prompt):
    if not base_url or not api_key or not model_name: return "Error: Custom provider configuration incomplete."
    try:
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Custom Provider Error: {str(e)}"

# ============================================
# 4. Multi-Agent Debate Logic (Sequential, Logic Correct)
# ============================================
def execute_debate(session_id, prompt, api_keys, custom_providers):
    def get_ai():
        if api_keys:
            p = list(api_keys.keys())[0]
            return p, api_keys[p][0], None
        elif custom_providers:
            cp = custom_providers[0]
            return "custom", cp['api_key'], cp
        return None, None, None

    provider, key, custom_info = get_ai()
    if not key:
        return {"error": "No AI agents available."}

    # 1. Lead Architect generates solution
    lead_prompt = f"You are a Senior Software Architect. Provide a comprehensive solution, code, and architecture plan for the following request.\n\nRequest: {prompt}"
    if provider == "custom":
        architect_solution = call_custom_ai(custom_info['base_url'], key, custom_info['model'], lead_prompt)
    else:
        architect_solution = call_ai(lead_prompt, provider, key)

    # 2. Skeptic Agent critiques the generated solution (Corrected Logic)
    skeptic_prompt = f"You are a Skeptic Agent. Your job is to critically review the solution provided by the Senior Software Architect. Identify edge cases, security flaws, potential bugs, and suggest improvements.\n\nSolution to review:\n{architect_solution}"
    if provider == "custom":
        critique = call_custom_ai(custom_info['base_url'], key, custom_info['model'], skeptic_prompt)
    else:
        critique = call_ai(skeptic_prompt, provider, key)

    # 3. Lead Architect refines based on Critique
    refine_prompt = f"You are a Senior Software Architect. You received the following critique on your solution. Refine your solution to address the critiques and provide the final optimal answer.\n\nYour Original Solution:\n{architect_solution}\n\nCritique:\n{critique}"
    if provider == "custom":
        final_answer = call_custom_ai(custom_info['base_url'], key, custom_info['model'], refine_prompt)
    else:
        final_answer = call_ai(refine_prompt, provider, key)

    return {"architect": architect_solution, "skeptic": critique, "final": final_answer}

# ============================================
# 5. Routes
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId')
    mode = data.get('mode', 'General')
    user_message = data.get('message', '')
    debate_on = data.get('debateOn', False)
    api_keys = data.get('apiKeys', {})
    custom_providers = data.get('customProviders', [])
    pg_state = data.get('pgState', {})

    if not user_message.strip():
        return jsonify({"error": "Empty message"}), 400

    # Session ID Logic
    if not session_id or session_id not in session_store:
        session_id = str(uuid.uuid4())
    
    if session_id not in session_store:
        session_store[session_id] = {"history": [], "mode": mode, "pg_state": pg_state, "created_at": datetime.now().isoformat()}
        save_sessions(session_store)

    session_store[session_id]["history"].append({"role": "user", "content": user_message})
    save_sessions(session_store)

    if mode == "General":
        base_prompt = "You are a helpful, friendly, and concise AI assistant. Answer clearly and directly."
        if api_keys:
            p = list(api_keys.keys())[0]
            ai_reply = call_ai(user_message, p, api_keys[p][0])
        elif custom_providers:
            cp = custom_providers[0]
            ai_reply = call_custom_ai(cp['base_url'], cp['api_key'], cp['model'], user_message)
        else:
            ai_reply = "Error: No API keys or Custom Providers found."
    else:
        prompt_with_context = f"User Goal: {user_message}\n\nContext: {pg_state.get('coreSpec', 'None defined yet')}"
        if not debate_on:
            result = execute_debate(session_id, prompt_with_context, api_keys, custom_providers)
            ai_reply = f"**Lead Architect's Solution:**\n{result['architect']}\n\n---\n**Skeptic's Critique:**\n{result['skeptic']}\n\n---\n**Final Refined Solution:**\n{result['final']}"
        else:
            result = execute_debate(session_id, prompt_with_context, api_keys, custom_providers)
            ai_reply = result['final']

    session_store[session_id]["history"].append({"role": "assistant", "content": ai_reply})
    save_sessions(session_store)

    return jsonify({"response": ai_reply, "mode": mode, "debateOn": debate_on, "sessionId": session_id})

@app.route('/api/export/<session_id>', methods=['GET'])
def export_chat(session_id):
    if session_id not in session_store:
        return jsonify({"error": "Session not found"}), 404
    
    session = session_store[session_id]
    history = session['history']
    markdown = f"# AI Architect Session Report\n\n**Session ID:** {session_id}\n**Created At:** {session['created_at']}\n**Mode:** {session['mode']}\n\n"
    markdown += "## Conversation History\n\n"
    for msg in history:
        role = "🤖 **Assistant**" if msg['role'] == 'assistant' else "👤 **User**"
        markdown += f"### {role}\n{msg['content']}\n\n---\n\n"
    memory_file = io.BytesIO()
    memory_file.write(markdown.encode('utf-8'))
    memory_file.seek(0)
    return send_file(memory_file, as_attachment=True, download_name=f"ai_architect_session_{session_id}.md", mimetype='text/markdown')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
