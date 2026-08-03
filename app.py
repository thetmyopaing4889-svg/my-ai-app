import os
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

BUILTIN_PROVIDERS = {
    "deepseek":   {"base_url": "https://api.deepseek.com",           "model": "deepseek-chat"},
    "github":     {"base_url": "https://models.github.ai/inference", "model": "gpt-4o-mini"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",       "model": "deepseek/deepseek-r1:free"},
}
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

# ============================================
# Backend မှာ Session Data ကို File မှာ မသိမ်းတော့ဘူး
# Frontend (IndexedDB) ကို Single Source of Truth အဖြစ် ထားမယ်
# Export လုပ်ချင်ရင် Frontend ကနေ Request ပို့ရုံပါပဲ
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

    lead_prompt = f"You are a Senior Software Architect. Provide a comprehensive solution, code, and architecture plan for the following request.\n\nRequest: {prompt}"
    if provider == "custom":
        architect_solution = call_custom_ai(custom_info['base_url'], key, custom_info['model'], lead_prompt)
    else:
        architect_solution = call_ai(lead_prompt, provider, key)

    skeptic_prompt = f"You are a Skeptic Agent. Your job is to critically review the solution provided by the Senior Software Architect. Identify edge cases, security flaws, potential bugs, and suggest improvements.\n\nSolution to review:\n{architect_solution}"
    if provider == "custom":
        critique = call_custom_ai(custom_info['base_url'], key, custom_info['model'], skeptic_prompt)
    else:
        critique = call_ai(skeptic_prompt, provider, key)

    refine_prompt = f"You are a Senior Software Architect. You received the following critique on your solution. Refine your solution to address the critiques and provide the final optimal answer.\n\nYour Original Solution:\n{architect_solution}\n\nCritique:\n{critique}"
    if provider == "custom":
        final_answer = call_custom_ai(custom_info['base_url'], key, custom_info['model'], refine_prompt)
    else:
        final_answer = call_ai(refine_prompt, provider, key)

    return {"architect": architect_solution, "skeptic": critique, "final": final_answer}

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

    # ✅ Session ID ကို Frontend ကနေပဲ ယူမယ်။
    # အကယ်၍ session_id မရှိရင် frontend က ထုတ်ထားတဲ့ format အတိုင်း ဖန်တီးပေးမယ်
    if not session_id:
        session_id = f"session_{int(datetime.now().timestamp())}"

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

    return jsonify({
        "response": ai_reply,
        "mode": mode,
        "debateOn": debate_on,
        "sessionId": session_id
    })

# ✅ Export Route ကို ပြန်ပြင်ပါ။
# Backend မှာ Session မသိမ်းတော့ဘူး။ ဒါကြောင့် Export လုပ်ချင်ရင် Frontend ကနေ History အကုန် ပို့ပေးရမယ်။
# ဒါပေမယ့် Manus AI ရဲ့ အဆိုပြုချက်အရ Frontend က IndexedDB ထဲမှာ ရှိနေပြီးသားမို့ Export ကို Frontend ကနေပဲ လုပ်သင့်တယ်။
# ဒါကြောင့် /api/export route ကို ဖယ်ရှားပြီး Frontend မှာ exportChat() function ကို ပြန်ရေးပါမယ်။

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
