import os
import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_CONTEXT_RESPONSE_LENGTH = 30_000


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
    if selector and selector.get("provider") and selector.get("key"):
        selected_provider = selector["provider"]
        selected_key = selector["key"]
        custom_name = (
            selected_provider[len("custom:"):]
            if selected_provider.startswith("custom:")
            else selected_provider
        )
        if selected_provider.startswith("custom:"):
            for custom_provider in custom_providers:
                if custom_provider.get("name") == custom_name:
                    if selected_key == custom_provider["api_key"]:
                        return "custom", selected_key, custom_provider
                    break
        if selected_provider in api_keys:
            keys = api_keys[selected_provider]
            if selected_key in keys:
                return selected_provider, selected_key, None
            # A stale dropdown value should not block a valid configured agent.
        for custom_provider in custom_providers:
            if custom_provider.get("name") == custom_name:
                return "custom", custom_provider["api_key"], custom_provider

    for provider, keys in api_keys.items():
        if keys:
            return provider, keys[0], None
    if custom_providers:
        custom_provider = custom_providers[0]
        return "custom", custom_provider["api_key"], custom_provider
    raise ValueError("No AI provider is configured.")


def configured_agents(api_keys, custom_providers):
    """Turn every configured API key/provider into an independently callable agent."""
    agents = []
    for provider, keys in api_keys.items():
        for index, key in enumerate(keys):
            agents.append({
                "id": f"{provider}-{index + 1}",
                "provider": provider,
                "key": key,
                "custom": None,
            })
    for index, custom_provider in enumerate(custom_providers):
        agents.append({
            "id": custom_provider.get("name") or f"custom-{index + 1}",
            "provider": "custom",
            "key": custom_provider["api_key"],
            "custom": custom_provider,
        })
    if not agents:
        raise ValueError("No AI provider is configured.")
    return agents


def clip_text(value, limit=MAX_CONTEXT_RESPONSE_LENGTH):
    value = str(value or "")
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n\n[Response truncated for comparison]"


def run_agent(agent, prompt):
    try:
        response = call_provider(
            prompt,
            agent["provider"],
            agent["key"],
            agent["custom"],
        )
        return {
            "id": agent["id"],
            "provider": agent["provider"],
            "response": clip_text(response),
            "error": None,
        }
    except (ValueError, AIServiceError) as error:
        return {
            "id": agent["id"],
            "provider": agent["provider"],
            "response": None,
            "error": str(error),
        }


def run_agents_parallel(agents, prompt):
    """Ask every configured agent concurrently so one slow provider does not serialize all calls."""
    results = []
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(run_agent, agent, prompt): agent for agent in agents}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result["id"])
    successful = [result for result in results if result["response"]]
    if not successful:
        errors = "; ".join(result["error"] or "Unknown provider error" for result in results)
        raise AIServiceError(f"All AI agents failed: {errors}")
    return results


def candidate_prompt(question, mode):
    return (
        "You are one of several independent AI agents answering the same user request. "
        "Give your strongest, accurate, practical answer. Do not mention this orchestration "
        "unless it is relevant to the user's request.\n\n"
        f"Mode: {mode}\n"
        f"User request:\n{question}"
    )


def format_agent_answers(results):
    sections = []
    for result in results:
        if result["response"]:
            sections.append(f"### {result['id']}\n{result['response']}")
        else:
            sections.append(f"### {result['id']}\n[Agent failed: {result['error']}]")
    return "\n\n---\n\n".join(sections)


def run_justice_recommendation(question, mode, results, justice_selector, api_keys, custom_providers):
    justice_provider, justice_key, justice_custom = selector_for(
        "justice", justice_selector, api_keys, custom_providers
    )
    candidate_answers = format_agent_answers(results)
    prompt = (
        "You are Justice AI, the impartial evaluator. Several AI agents answered the same "
        "user request below. Compare their answers for correctness, completeness, safety, "
        "and usefulness. Recommend the single best answer and provide that recommended answer "
        "clearly. Do not merely name an agent; explain the important reason briefly, then give "
        "the best answer the user should use.\n\n"
        f"User request:\n{question}\n\n"
        f"Mode: {mode}\n\n"
        f"Candidate answers:\n{candidate_answers}"
    )
    try:
        recommendation = call_provider(
            prompt,
            justice_provider,
            justice_key,
            justice_custom,
        )
    except (ValueError, AIServiceError):
        recommendation = (
            "[Justice AI could not complete the recommendation. "
            "The candidate answers above are still available for review.]"
        )
    return recommendation, candidate_answers


def run_peer_debate(question, mode, results, agents, chair_selector, api_keys, custom_providers):
    candidate_answers = format_agent_answers(results)
    debate_prompt = (
        "You are participating in a peer debate with other AI agents. Review all candidate "
        "answers below against the original user request. Identify mistakes and missing details, "
        "then propose the strongest corrected answer. Do not defer to a named judge; reason from "
        "the evidence and produce a concrete improved answer.\n\n"
        f"Original user request:\n{question}\n\n"
        f"Mode: {mode}\n\n"
        f"Candidate answers:\n{candidate_answers}"
    )
    try:
        debate_results = run_agents_parallel(agents, debate_prompt)
        debate_answers = format_agent_answers(debate_results)
    except AIServiceError:
        debate_results = []
        debate_answers = (
            "[Peer review could not be completed. "
            "The initial candidate answers are still available for review.]"
        )

    # A configured agent chairs the final consensus round; Justice is deliberately not used.
    chair_provider, chair_key, chair_custom = selector_for(
        "pg", chair_selector, api_keys, custom_providers
    )
    final_prompt = (
        "You are the chair of a peer AI debate. Produce one final, direct answer to the "
        "original user request by synthesizing the peer reviews below. Resolve disagreements "
        "using correctness and usefulness. Return only the final answer and do not mention "
        "internal orchestration, agents, or voting.\n\n"
        f"Original user request:\n{question}\n\n"
        f"Peer reviews:\n{debate_answers}"
    )
    successful_reviews = [result["response"] for result in debate_results if result["response"]]
    if not successful_reviews:
        final_answer = (
            "[Peer debate final synthesis was unavailable. "
            "Review the initial candidate answers above.]"
        )
    else:
        try:
            final_answer = call_provider(
                final_prompt,
                chair_provider,
                chair_key,
                chair_custom,
            )
        except (ValueError, AIServiceError):
            final_answer = (
                "[The selected PG chair could not complete the final synthesis. "
                "The strongest available peer review is shown below.]"
                f"\n\n{successful_reviews[0]}"
            )
    return final_answer, candidate_answers, debate_answers


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
    return genai.Client(
        api_key=api_key,
        http_options=genai.types.HttpOptions(timeout=UPSTREAM_TIMEOUT_SECONDS * 1000),
    )

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
        agents = configured_agents(api_keys, custom_providers)
        initial_results = run_agents_parallel(
            agents,
            candidate_prompt(
                user_message if mode == "General" else (
                    f"{user_message}\n\nProject context: "
                    f"{pg_state.get('coreSpec', 'None defined yet')}"
                ),
                mode,
            ),
        )
        question_for_judging = user_message
        if mode == "Professional":
            question_for_judging = (
                f"{user_message}\n\nProject context: "
                f"{pg_state.get('coreSpec', 'None defined yet')}"
            )

        if mode != "Professional" or not debate_on:
            recommendation, candidate_answers = run_justice_recommendation(
                question_for_judging,
                mode,
                initial_results,
                data.get("justiceSelector"),
                api_keys,
                custom_providers,
            )
            ai_reply = (
                "## AI Responses\n\n"
                f"{candidate_answers}\n\n"
                "---\n\n"
                "## Justice AI Recommendation\n\n"
                f"{recommendation}"
            )
        else:
            final_answer, candidate_answers, debate_answers = run_peer_debate(
                question_for_judging,
                mode,
                initial_results,
                agents,
                data.get("pgSelector"),
                api_keys,
                custom_providers,
            )
            ai_reply = (
                "## Peer Debate Result\n\n"
                f"{final_answer}\n\n"
                "---\n\n"
                "## Peer Review Details\n\n"
                f"{debate_answers}"
            )
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
