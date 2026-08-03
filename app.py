import os
import ipaddress
import socket
from urllib.parse import urlparse
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import google.genai as genai
import requests

app = Flask(__name__)
CORS(app)

BUILTIN_PROVIDERS = {
    "deepseek":   {"base_url": "https://api.deepseek.com",           "model": "deepseek-chat"},
    "github":     {"base_url": "https://models.github.ai/inference", "model": "gpt-4o-mini"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",       "model": "deepseek/deepseek-r1:free"},
}
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
MAX_MESSAGE_LENGTH = 100_000
UPSTREAM_TIMEOUT_SECONDS = 30


class AIServiceError(Exception):
    """An expected upstream/provider failure that should not become a 500."""


def error_response(message, status):
    return jsonify({"error": message}), status


def validate_request_data(data):
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message is too long (maximum {MAX_MESSAGE_LENGTH} characters).")

    mode = data.get("mode", "General")
    if mode not in {"General", "Professional"}:
        raise ValueError("Mode must be General or Professional.")

    if not isinstance(data.get("debateOn", False), bool):
        raise ValueError("debateOn must be a boolean.")

    api_keys = data.get("apiKeys", {})
    if not isinstance(api_keys, dict):
        raise ValueError("apiKeys must be an object.")
    for provider, keys in api_keys.items():
        if provider not in BUILTIN_PROVIDERS and provider != "gemini":
            raise ValueError(f"Unsupported provider: {provider}.")
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError(f"API keys for {provider} must be a non-empty string array.")

    custom_providers = data.get("customProviders", [])
    if not isinstance(custom_providers, list):
        raise ValueError("customProviders must be an array.")
    for provider in custom_providers:
        if not isinstance(provider, dict):
            raise ValueError("Each custom provider must be an object.")
        if any(not isinstance(provider.get(field), str) or not provider[field].strip()
               for field in ("base_url", "api_key", "model")):
            raise ValueError("Each custom provider needs base_url, api_key, and model.")

    pg_state = data.get("pgState", {})
    if not isinstance(pg_state, dict):
        raise ValueError("pgState must be an object.")

    for selector_name in ("pgSelector", "justiceSelector"):
        selector = data.get(selector_name)
        if selector is not None and not isinstance(selector, dict):
            raise ValueError(f"{selector_name} must be an object.")

    return message, mode, data.get("debateOn", False), api_keys, custom_providers, pg_state


def selector_for(provider_name, selector, api_keys, custom_providers):
    """Resolve a selected built-in/custom provider, with a safe legacy fallback."""
    if selector and selector.get("provider"):
        selected_provider = selector["provider"]
        selected_key = selector.get("key")
        if selected_provider in api_keys:
            keys = api_keys[selected_provider]
            if selected_key in keys:
                return selected_provider, selected_key, None
            raise ValueError(f"No matching key selected for {selected_provider}.")
        for custom_provider in custom_providers:
            if custom_provider.get("name") == selected_provider:
                return "custom", custom_provider["api_key"], custom_provider
        raise ValueError(f"Selected provider is not configured: {selected_provider}.")

    for provider, keys in api_keys.items():
        if keys:
            return provider, keys[0], None
    if custom_providers:
        custom_provider = custom_providers[0]
        return "custom", custom_provider["api_key"], custom_provider
    raise ValueError("No AI provider is configured.")


def validate_custom_url(base_url):
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Custom provider URL must be a public HTTPS URL.")

    hostname = parsed.hostname
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except (OSError, ValueError):
        raise ValueError("Custom provider hostname could not be resolved.")

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Custom provider URL cannot point to a private or local address.")


def call_provider(prompt, provider, api_key, custom_info=None, custom_model=None, timeout=UPSTREAM_TIMEOUT_SECONDS):
    if provider == "custom":
        return call_custom_ai(custom_info["base_url"], api_key, custom_info["model"], prompt, timeout)
    return call_ai(prompt, provider, api_key, custom_model=custom_model, timeout=timeout)

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

def call_ai(prompt, provider, api_key, custom_model=None, timeout=UPSTREAM_TIMEOUT_SECONDS):
    if not api_key:
        raise ValueError(f"No API key provided for {provider}.")
    try:
        if provider == "gemini":
            client = get_gemini_client(api_key)
            model_name = custom_model or GEMINI_DEFAULT_MODEL
            resp = client.models.generate_content(model=model_name, contents=prompt)
            if not resp.text:
                raise AIServiceError(f"{provider} returned an empty response.")
            return resp.text.strip()
        else:
            client = get_openai_client(provider, api_key)
            config = BUILTIN_PROVIDERS.get(provider)
            if not client or not config:
                raise ValueError(f"{provider} configuration issue.")
            model_name = custom_model or config["model"]
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            content = resp.choices[0].message.content
            if not content:
                raise AIServiceError(f"{provider} returned an empty response.")
            return content
    except Exception as e:
        if isinstance(e, (ValueError, AIServiceError)):
            raise
        raise AIServiceError(f"{provider} request failed.") from e

def call_custom_ai(base_url, api_key, model_name, prompt, timeout=UPSTREAM_TIMEOUT_SECONDS):
    if not base_url or not api_key or not model_name:
        raise ValueError("Custom provider configuration incomplete.")
    try:
        validate_custom_url(base_url)
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not content:
            raise AIServiceError("Custom provider returned an empty response.")
        return content
    except Exception as e:
        if isinstance(e, (ValueError, AIServiceError)):
            raise
        raise AIServiceError("Custom provider request failed.") from e

def execute_debate(prompt, api_keys, custom_providers, pg_selector=None, justice_selector=None):
    architect_provider, architect_key, architect_custom = selector_for(
        "pg", pg_selector, api_keys, custom_providers
    )
    lead_prompt = f"You are a Senior Software Architect. Provide a comprehensive solution, code, and architecture plan for the following request.\n\nRequest: {prompt}"
    architect_solution = call_provider(
        lead_prompt, architect_provider, architect_key, architect_custom
    )

    skeptic_provider, skeptic_key, skeptic_custom = selector_for(
        "justice", justice_selector, api_keys, custom_providers
    )
    skeptic_prompt = f"You are a Skeptic Agent. Your job is to critically review the solution provided by the Senior Software Architect. Identify edge cases, security flaws, potential bugs, and suggest improvements.\n\nSolution to review:\n{architect_solution}"
    critique = call_provider(skeptic_prompt, skeptic_provider, skeptic_key, skeptic_custom)

    refine_prompt = f"You are a Senior Software Architect. You received the following critique on your solution. Refine your solution to address the critiques and provide the final optimal answer.\n\nYour Original Solution:\n{architect_solution}\n\nCritique:\n{critique}"
    final_answer = call_provider(
        refine_prompt, architect_provider, architect_key, architect_custom
    )

    return {"architect": architect_solution, "skeptic": critique, "final": final_answer}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"})


@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return error_response("Request must use application/json.", 400)
    data = request.get_json(silent=True)
    try:
        user_message, mode, debate_on, api_keys, custom_providers, pg_state = validate_request_data(data)
    except ValueError as e:
        return error_response(str(e), 400)

    session_id = data.get('sessionId')
    if session_id is not None and (not isinstance(session_id, str) or len(session_id) > 200):
        return error_response("sessionId must be a short string.", 400)

    if not session_id:
        session_id = f"session_{int(datetime.now().timestamp())}"

    try:
        if mode == "General":
            base_prompt = "You are a helpful, friendly, and concise AI assistant. Answer clearly and directly."
            prompt = f"{base_prompt}\n\nUser: {user_message}"
            provider, key, custom_info = selector_for("general", None, api_keys, custom_providers)
            ai_reply = call_provider(prompt, provider, key, custom_info)
        else:
            prompt_with_context = f"User Goal: {user_message}\n\nContext: {pg_state.get('coreSpec', 'None defined yet')}"
            if debate_on:
                result = execute_debate(
                    prompt_with_context,
                    api_keys,
                    custom_providers,
                    data.get("pgSelector"),
                    data.get("justiceSelector"),
                )
                ai_reply = f"**Lead Architect's Solution:**\n{result['architect']}\n\n---\n**Skeptic's Critique:**\n{result['skeptic']}\n\n---\n**Final Refined Solution:**\n{result['final']}"
            else:
                provider, key, custom_info = selector_for(
                    "pg", data.get("pgSelector"), api_keys, custom_providers
                )
                ai_reply = call_provider(prompt_with_context, provider, key, custom_info)
    except ValueError as e:
        return error_response(str(e), 400)
    except AIServiceError as e:
        return error_response(str(e), 502)

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
