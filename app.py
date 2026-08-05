import os
import ipaddress
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import google.genai as genai
import requests

app = Flask(__name__)
CORS(app)

BUILTIN_PROVIDERS = {
    "deepseek":   {"base_url": "https://api.deepseek.com",           "model": "deepseek-chat"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",     "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",       "model": "deepseek/deepseek-chat"},
}
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
MAX_MESSAGE_LENGTH = 100_000
UPSTREAM_TIMEOUT_SECONDS = 30
MAX_CONTEXT_RESPONSE_LENGTH = 30_000
MAX_PG_SUMMARY_LENGTH = 12_000


class AIServiceError(Exception):
    """An expected upstream/provider failure that should not become a 500."""


def error_response(message, status):
    return jsonify({"error": message}), status


def explain_provider_error(provider, error):
    """Return a useful provider diagnosis without exposing keys or raw upstream text."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    if status_code in {401, 403}:
        return f"{provider}: API key is invalid, expired, or not permitted (HTTP {status_code})."
    if status_code == 404:
        return f"{provider}: model or API endpoint was not found (HTTP 404)."
    if status_code == 429:
        return f"{provider}: rate limit or account quota was reached (HTTP 429)."
    if provider == "gemini" and status_code == 400:
        return f"{provider}: API key is invalid or Google rejected the request (HTTP 400)."
    if status_code is not None and 400 <= status_code < 500:
        return f"{provider}: provider rejected the request (HTTP {status_code})."
    if status_code is not None and status_code >= 500:
        return f"{provider}: provider server error (HTTP {status_code}). Try again shortly."

    error_text = str(error).lower()
    if any(term in error_text for term in ("timeout", "timed out", "deadline exceeded")):
        return f"{provider}: provider request timed out."
    if any(term in error_text for term in ("connection", "dns", "network", "unreachable")):
        return f"{provider}: could not connect to the provider."
    if any(term in error_text for term in ("invalid api key", "authentication", "unauthorized")):
        return f"{provider}: API key authentication failed."
    if any(term in error_text for term in ("quota", "rate limit", "resource exhausted")):
        return f"{provider}: account quota or rate limit was reached."
    if any(term in error_text for term in ("model", "not found")):
        return f"{provider}: configured model or endpoint was rejected."
    return f"{provider}: provider returned an unexpected error ({type(error).__name__})."


def validate_provider_config(api_keys, custom_providers):
    if not isinstance(api_keys, dict):
        raise ValueError("apiKeys must be an object.")
    for provider, keys in api_keys.items():
        if provider not in BUILTIN_PROVIDERS and provider != "gemini":
            raise ValueError(f"Unsupported provider: {provider}.")
        if not isinstance(keys, list) or not keys or any(
            not isinstance(key, str) or not key.strip() for key in keys
        ):
            raise ValueError(f"API keys for {provider} must be a non-empty string array.")

    if not isinstance(custom_providers, list):
        raise ValueError("customProviders must be an array.")
    for provider in custom_providers:
        if not isinstance(provider, dict):
            raise ValueError("Each custom provider must be an object.")
        if any(
            not isinstance(provider.get(field), str) or not provider[field].strip()
            for field in ("name", "base_url", "api_key", "model")
        ):
            raise ValueError("Each custom provider needs name, base_url, api_key, and model.")


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
    custom_providers = data.get("customProviders", [])
    validate_provider_config(api_keys, custom_providers)

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


PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "custom": "Custom AI",
}


def provider_label(provider):
    return PROVIDER_LABELS.get(provider, provider.title())


def agent_display_name(result):
    provider = result.get("provider", "custom")
    agent_id = result.get("id", "")
    if provider == "custom":
        return agent_id or "Custom AI"
    suffix = agent_id.rsplit("-", 1)[-1] if "-" in agent_id else ""
    return f"{provider_label(provider)} {suffix}" if suffix.isdigit() else provider_label(provider)


def agent_card_data(result):
    is_connected = bool(result.get("response"))
    provider = result.get("provider", "custom")
    return {
        "id": result.get("id"),
        "name": agent_display_name(result),
        "provider": provider,
        "status": "connected" if is_connected else "unavailable",
        "statusPercent": 100 if is_connected else 0,
        "statusLabel": "Response ready" if is_connected else "Request failed",
        "credit": {
            "label": "Credit" if provider == "deepseek" else "Quota",
            "value": "Open Settings → Check connections" if provider == "deepseek" else "Not exposed by provider API",
            "percent": None,
        },
        "response": result.get("response"),
        "error": result.get("error"),
    }


def selected_agent_card(name, provider, response):
    return {
        "id": name.lower().replace(" ", "-"),
        "name": f"{name} · {provider_label(provider)}",
        "provider": provider,
        "status": "connected",
        "statusPercent": 100,
        "statusLabel": "Response ready",
        "credit": {
            "label": "Credit" if provider == "deepseek" else "Quota",
            "value": "Open Settings → Check connections" if provider == "deepseek" else "Not exposed by provider API",
            "percent": None,
        },
        "response": response,
        "error": None,
    }


def format_project_context(pg_state):
    pg_state = pg_state if isinstance(pg_state, dict) else {}
    core_spec = str(pg_state.get("coreSpec") or "None defined yet").strip()
    active_focus = str(pg_state.get("activeFocus") or "Not set").strip()
    progress = pg_state.get("progress") or []
    if not isinstance(progress, list):
        progress = []
    completed_phases = [
        str(phase).strip() for phase in progress
        if isinstance(phase, str) and phase.strip()
    ]
    completed_text = "\n".join(f"- {phase}" for phase in completed_phases) or "- None yet"
    lock_status = "LOCKED — do not redefine the scope or plan." if pg_state.get("locked") else "UNLOCKED — the user may revise the plan."
    return (
        f"Lock status: {lock_status}\n"
        f"Core specification:\n{clip_text(core_spec, MAX_PG_SUMMARY_LENGTH)}\n"
        f"Completed phases:\n{completed_text}\n"
        f"Active focus:\n{active_focus}"
    )


def summarize_project_memory(message, current_summary, pg_selector, api_keys, custom_providers):
    provider, key, custom = selector_for("pg", pg_selector, api_keys, custom_providers)
    prompt = (
        "You are Project Guardian for a personal AI coding workspace. Convert the user's "
        "latest message into a concise English project-memory update. Preserve only durable "
        "facts: project goals, architecture decisions, constraints, completed work, and "
        "next steps. Merge it with the existing memory, remove obsolete contradictions, and "
        "return only the updated summary in plain text. Do not include secrets, API keys, "
        "conversation chatter, or explanations about this instruction.\n\n"
        f"Existing project memory:\n{current_summary or '(none)'}\n\n"
        f"Latest user message:\n{message}"
    )
    summary = call_provider(prompt, provider, key, custom)
    summary = clip_text(summary, MAX_PG_SUMMARY_LENGTH).strip()
    if not summary:
        raise AIServiceError("Project Guardian returned an empty summary.")
    return summary


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


def check_provider_status(agent):
    """Check authentication/connectivity without exposing keys or claiming a balance."""
    try:
        if agent["provider"] == "custom":
            validate_custom_url(agent["custom"]["base_url"])
            return {
                "id": agent["id"],
                "provider": agent["provider"],
                "status": "configured",
                "detail": "Custom provider saved. Live quota is not exposed by this app.",
                "credit": {"label": "Quota", "value": "Check provider dashboard", "percent": None},
            }
        if agent["provider"] == "deepseek":
            response = requests.get(
                "https://api.deepseek.com/user/balance",
                headers={"Authorization": f"Bearer {agent['key']}", "Accept": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            balance_infos = response.json().get("balance_infos", [])
            balances = []
            for balance in balance_infos:
                currency = balance.get("currency")
                total = balance.get("total_balance")
                if currency and total is not None:
                    balances.append(f"{currency} {total}")
            return {
                "id": agent["id"],
                "provider": agent["provider"],
                "status": "connected",
                "detail": (
                    f"Balance: {', '.join(balances)}."
                    if balances else
                    "Connected, but no balance amount was returned."
                ),
                "credit": {
                    "label": "Credit",
                    "value": ", ".join(balances) if balances else "No balance returned",
                    "percent": None,
                },
            }
        if agent["provider"] == "gemini":
            # Use Google's key-authenticated REST endpoint. The SDK's models.list()
            # can fail with an opaque RuntimeError and is not a reliable key check.
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": agent["key"]},
                timeout=10,
            )
            response.raise_for_status()
        elif agent["provider"] == "openrouter":
            # OpenRouter's public /models endpoint does not validate credentials.
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={
                    "Authorization": f"Bearer {agent['key']}",
                    "Accept": "application/json",
                },
                timeout=10,
            )
            response.raise_for_status()
        else:
            client = get_openai_client(agent["provider"], agent["key"])
            client.models.list()
        return {
            "id": agent["id"],
            "provider": agent["provider"],
            "status": "connected",
            "detail": (
                "Connected. Check the provider dashboard for quota and billing."
                if agent["provider"] != "openrouter"
                else "Connected. Credit balance requires an OpenRouter management key."
            ),
            "credit": {
                "label": "Quota",
                "value": "OpenRouter management key required" if agent["provider"] == "openrouter" else "Check provider dashboard",
                "percent": None,
            },
        }
    except Exception as error:
        return {
            "id": agent["id"],
            "provider": agent["provider"],
            "status": "unavailable",
            "detail": explain_provider_error(agent["provider"], error),
            "credit": {"label": "Quota", "value": "Unavailable", "percent": None},
        }


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
    return final_answer, candidate_answers, debate_answers, debate_results


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
        raise AIServiceError(explain_provider_error(provider, e)) from e

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
        raise AIServiceError(explain_provider_error("custom provider", e)) from e

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manifest.json')
def manifest():
    return send_from_directory(
        app.root_path,
        'manifest.json',
        mimetype='application/manifest+json',
        max_age=3600,
    )


@app.route('/sw.js')
def service_worker():
    return send_from_directory(
        app.root_path,
        'sw.js',
        mimetype='application/javascript',
        max_age=0,
    )


@app.route('/icons/<path:filename>')
def app_icon(filename):
    return send_from_directory(
        os.path.join(app.root_path, 'icons'),
        filename,
        max_age=86400,
    )


@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"})


@app.route('/api/pg-summary', methods=['POST'])
def pg_summary():
    if not request.is_json:
        return error_response("Request must use application/json.", 400)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Request body must be a JSON object.", 400)

    message = data.get("message")
    current_summary = data.get("currentSummary", "")
    locked = data.get("locked", False)
    if not isinstance(message, str) or not message.strip():
        return error_response("Message must be a non-empty string.", 400)
    if len(message) > MAX_MESSAGE_LENGTH:
        return error_response(f"Message is too long (maximum {MAX_MESSAGE_LENGTH} characters).", 400)
    if not isinstance(current_summary, str):
        return error_response("currentSummary must be a string.", 400)
    if not isinstance(locked, bool):
        return error_response("locked must be a boolean.", 400)
    if locked:
        return jsonify({"summary": clip_text(current_summary, MAX_PG_SUMMARY_LENGTH), "locked": True})

    api_keys = data.get("apiKeys", {})
    custom_providers = data.get("customProviders", [])
    selector = data.get("pgSelector")
    if selector is not None and not isinstance(selector, dict):
        return error_response("pgSelector must be an object.", 400)
    try:
        validate_provider_config(api_keys, custom_providers)
        summary = summarize_project_memory(
            message,
            clip_text(current_summary, MAX_PG_SUMMARY_LENGTH),
            selector,
            api_keys,
            custom_providers,
        )
        return jsonify({"summary": summary})
    except ValueError as error:
        return error_response(str(error), 400)
    except AIServiceError as error:
        return error_response(str(error), 502)


@app.route('/api/provider-status', methods=['POST'])
def provider_status():
    if not request.is_json:
        return error_response("Request must use application/json.", 400)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Request body must be a JSON object.", 400)
    api_keys = data.get("apiKeys", {})
    custom_providers = data.get("customProviders", [])
    try:
        validate_provider_config(api_keys, custom_providers)
        agents = configured_agents(api_keys, custom_providers)
        return jsonify({
            "providers": [check_provider_status(agent) for agent in agents],
            "creditNote": "Providers do not expose a reliable shared credit balance here; check their dashboards for exact quota.",
        })
    except ValueError as error:
        return error_response(str(error), 400)


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
        session_id = f"session_{uuid.uuid4().hex}"

    try:
        agents = configured_agents(api_keys, custom_providers)
        project_context = format_project_context(pg_state)
        initial_results = run_agents_parallel(
            agents,
            candidate_prompt(
                user_message if mode == "General" else (
                    f"{user_message}\n\nProject context: "
                    f"{project_context}"
                ),
                mode,
            ),
        )
        question_for_judging = user_message
        if mode == "Professional":
            question_for_judging = (
                f"{user_message}\n\nProject context: "
                f"{project_context}"
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
            justice_provider, _, _ = selector_for(
                "justice",
                data.get("justiceSelector"),
                api_keys,
                custom_providers,
            )
            response_payload = {
                "agentAnswers": [agent_card_data(result) for result in initial_results],
                "recommendation": selected_agent_card("Justice AI", justice_provider, recommendation),
            }
        else:
            final_answer, candidate_answers, debate_answers, debate_results = run_peer_debate(
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
            chair_provider, _, _ = selector_for(
                "pg",
                data.get("pgSelector"),
                api_keys,
                custom_providers,
            )
            response_payload = {
                "agentAnswers": [agent_card_data(result) for result in initial_results],
                "peerReviews": [agent_card_data(result) for result in debate_results],
                "recommendation": selected_agent_card("Final Chair", chair_provider, final_answer),
            }
    except ValueError as e:
        return error_response(str(e), 400)
    except AIServiceError as e:
        return error_response(str(e), 502)

    return jsonify({
        "response": ai_reply,
        "mode": mode,
        "debateOn": debate_on,
        "sessionId": session_id,
        **response_payload,
    })

# ✅ Export Route ကို ပြန်ပြင်ပါ။
# Backend မှာ Session မသိမ်းတော့ဘူး။ ဒါကြောင့် Export လုပ်ချင်ရင် Frontend ကနေ History အကုန် ပို့ပေးရမယ်။
# ဒါပေမယ့် Manus AI ရဲ့ အဆိုပြုချက်အရ Frontend က IndexedDB ထဲမှာ ရှိနေပြီးသားမို့ Export ကို Frontend ကနေပဲ လုပ်သင့်တယ်။
# ဒါကြောင့် /api/export route ကို ဖယ်ရှားပြီး Frontend မှာ exportChat() function ကို ပြန်ရေးပါမယ်။

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
