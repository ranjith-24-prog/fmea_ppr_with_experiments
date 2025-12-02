# llm.py
import os, time, json, requests, re
from typing import Any, Dict, Tuple, Optional, List

from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------------
# Utilities
# ---------------------------

def _sanitize_common(text: str) -> str:
    text = text.replace('“','"').replace('”','"').replace('‟','"')
    text = text.replace("’","'").replace("‘","'")
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', ' ', text)
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text

def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()

# ---------------------------
# Model Registry
# ---------------------------
# Each entry defines: type, label, model, env, base_url
LLM_REGISTRY: Dict[str, Dict[str, str]] = {
    # 1) Perplexity (default)
    "perplexity/sonar-pro": {
        "label": "Perplexity Sonar Pro",
        "type": "perplexity",
        "model": "sonar-pro",
        "env": "PERPLEXITY_API_KEY",
        "base_url": os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai"),
    },

    # 2) OpenAI GPT-4o
    "openai/gpt-4o": {
        "label": "OpenAI GPT-4o",
        "type": "openai",
        "model": "gpt-4o",
        "env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
    },

    # 3) Anthropic Claude 3.5 Sonnet
    "anthropic/claude-3-5-sonnet": {
        "label": "Anthropic Claude 3.5 Sonnet",
        "type": "anthropic",
        "model": "claude-3-5-sonnet-20240620",
        "env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
    },

    # 4) Google Gemini 1.5 Pro
    "google/gemini-1.5-pro": {
        "label": "Google Gemini 1.5 Pro",
        "type": "google",
        "model": "gemini-1.5-pro",
        "env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
    },

    # 5) Mistral Large
    "mistral/mistral-large": {
        "label": "Mistral Large",
        "type": "mistral",
        "model": "mistral-large-latest",
        "env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1/chat/completions",
    },
}


DEFAULT_MODEL_ID = "perplexity/sonar-pro"

def _get_model_cfg(model_name_or_id: Optional[str]) -> Dict[str, Any]:
    """
    Accept either 'sonar-pro', 'perplexity/sonar-pro', or any registry key.
    """
    # Direct registry key
    if model_name_or_id in LLM_REGISTRY:
        mid = model_name_or_id
    else:
        # Map short names to a default entry
        short = (model_name_or_id or "").strip().lower()
        if short in ["sonar-pro", "perplexity", "pplx"]:
            mid = "perplexity/sonar-pro"
        elif short in ["gpt-4o", "openai"]:
            mid = "openai/gpt-4o"
        elif short in ["gpt-4o-mini"]:
            mid = "openai/gpt-4o-mini"
        elif short in ["claude", "claude-3.5", "claude-3-5-sonnet"]:
            mid = "anthropic/claude-3-5-sonnet"
        elif short in ["gemini", "gemini-1.5-pro"]:
            mid = "google/gemini-1.5-pro"
        elif short in ["mistral", "mistral-large"]:
            mid = "mistral/mistral-large"
        elif short in ["llama", "llama-3.1-70b"]:
            mid = "together/llama-3.1-70b"
        elif short in ["command-r-plus", "cohere"]:
            mid = "cohere/command-r-plus"
        elif short in ["groq", "groq-llama-3.1-70b"]:
            mid = "groq/llama-3.1-70b"
        else:
            mid = DEFAULT_MODEL_ID

    cfg = dict(LLM_REGISTRY[mid])
    cfg["id"] = mid
    cfg["api_key"] = os.getenv(cfg["env"], "")
    return cfg

# ---------------------------
# Error
# ---------------------------

class LLMError(Exception):
    pass

class LLLM:
    pass  # Keep for compatibility if referenced elsewhere

# ---------------------------
# Unified chat adapter
# ---------------------------

def run_llm_chat(model_id: str, messages: List[Dict[str, str]], temperature: float = 0.2, top_p: float = 0.9, max_tokens: int = 1400) -> Dict[str, Any]:
    """
    Run a chat completion against the selected provider.
    Returns { 'text': str, 'raw': dict, 'latency_ms': int }.
    'messages' is OpenAI-style list of {role, content}.
    """
    cfg = _get_model_cfg(model_id)
    if not cfg["api_key"]:
        raise LLMError(f"Missing API key for {cfg['label']} in env {cfg['env']}")

    t0 = time.time()
    t = cfg["type"]
    out_text = ""
    raw = {}

    if t in ["openai", "groq"]:
        url = cfg["base_url"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    elif t == "anthropic":
        url = cfg["base_url"]
        headers = {
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        sys_txt = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        user_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        conv = []
        for m in user_msgs:
            conv.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
        payload = {
            "model": cfg["model"],
            "system": sys_txt or None,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conv
        }
        raw = _post_json(url, headers, payload)
        parts = []
        for c in raw.get("content", []):
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
        out_text = "\n".join(parts)

    elif t == "google":
        # Gemini: compose a single prompt string
        api_key = cfg["api_key"]
        model = cfg["model"]
        url = f"{cfg['base_url']}/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "topP": top_p, "maxOutputTokens": max_tokens}
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    elif t == "perplexity":
        # Perplexity OpenAI-compatible, but base URL may differ
        url = (cfg["base_url"].rstrip("/") + "/chat/completions")
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    elif t == "mistral":
        url = cfg["base_url"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    elif t == "together":
        url = cfg["base_url"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    elif t == "cohere":
        url = cfg["base_url"]
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        payload = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        raw = _post_json(url, headers, payload)
        out_text = raw.get("text", "") or raw.get("reply", "") or raw.get("message", {}).get("content", "")

    else:
        raise LLMError(f"Unknown provider type: {t}")

    latency_ms = int((time.time() - t0) * 1000)
    return {"text": out_text, "raw": raw, "latency_ms": latency_ms, "cfg": cfg}

# ---------------------------
# Public client (API-compatible with your app)
# ---------------------------

print("[LLM] Registry ready. Default model:", DEFAULT_MODEL_ID)

class LLM:
    def __init__(self, model_name: str = "sonar-pro"):
        # Map incoming single-name to registry id
        cfg = _get_model_cfg(model_name)
        self.model_id = cfg["id"]
        print("[LLM] Initialized with:", cfg["label"], "| API key present:", bool(cfg["api_key"]))

    def set_model(self, model_name_or_id: str):
        cfg = _get_model_cfg(model_name_or_id)
        self.model_id = cfg["id"]
        print("[LLM] Switched to:", cfg["label"])

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 3200) -> str:
        out = run_llm_chat(
            self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )
        return out["text"]

    # Backwards-compatible methods

    def generate_ppr_from_fmea(self, fmea_rows_or_str):
        system_prompt = (
            "You are an expert in manufacturing and Failure Mode and Effect Analysis (FMEA). "
            "Given the following FMEA entries, identify the specific Products, Processes, and Resources referenced. "
            "Return a JSON object with keys 'products', 'processes', and 'resources', each with a list of unique names. "
            "If unknown, return empty list for that key. Only output the JSON."
        )
        if isinstance(fmea_rows_or_str, str):
            fmea_text = fmea_rows_or_str
        else:
            fmea_text = json.dumps(fmea_rows_or_str, ensure_ascii=False, indent=2)
        user_prompt = f"FMEA data:\n{fmea_text}\n\nReturn the categories JSON:"
        response = self._chat(system_prompt, user_prompt)
        try:
            return json.loads(response)
        except Exception as e:
            raise Exception(f"LLM return decoding failed: {e}. Response was: {response}")

    def generate_fmea_and_ppr_json(self, context_text: str, ppr_hint: dict = None):
        system = (
            "You are an expert in manufacturing Process FMEA (DIN EN 60812/APIS) and PPR (Product-Process-Resource) categorization.\n"
            "Given a production description, generate EXACTLY one JSON object with keys 'fmea' and 'ppr'.\n"
            "'fmea' should be an array of FMEA row objects with keys (at least 10 rows):\n"
            "Each FMEA row's 'system_element' must be a process step (e.g., Preparation, Welding, Inspection, Handling, Fixturing, Cleaning, Post-weld Inspection)\n"
            '["system_element", "function", "potential_failure", "c1", "potential_effect", "s1", "c2", "c3", '
            '"potential_cause", "o1", "current_preventive_action", "current_detection_action", "d1", "rpn1", '
            '"recommended_action", "rd", "action_taken", "s2", "o2", "d2", "rpn2", "notes"]\n'
            "'ppr' should be an object with keys 'products', 'processes', 'resources', each mapping to a list of strings.\n"
            "Output ONLY the JSON object, no markdown or extra text.\n"
            "Example:\n"
            '{\n'
            '  \"fmea\": [ ... FMEA rows ... ],\n'
            '  \"ppr\": {\n'
            '    \"products\": [\"Aluminium profile\"],\n'
            '    \"processes\": [\"Laser welding\"],\n'
            '    \"resources\": [\"Shielding gas\"]\n'
            '  }\n'
            '}'
        )
        hint_json = json.dumps(ppr_hint or {}, ensure_ascii=False)
        user = (
            f"Production scenario:\n{context_text}\n"
            f"PPR context hints:\n{hint_json}\n"
            "Remember: output exactly one JSON object as described above."
        )

        content = self._chat(system, user, temperature=0.2, max_tokens=3200)

        # Debug log to your terminal
        print("generate_fmea_and_ppr_json: raw LLM content (first 500 chars):")
        print(repr((content or "")[:500]))

        if not content or not str(content).strip():
            raise ValueError("LLM did not return any content (empty response).")

        # Primary parse
        try:
            data = json.loads(content)
        except Exception:
            # Clean obvious noise (control chars etc.)
            clean = _sanitize_common(content)

            # Try direct parse on cleaned text
            try:
                data = json.loads(clean)
            except Exception:
                # As a fallback, try to extract the first JSON object from the text
                text = clean.strip()
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = text[start : end + 1]
                    try:
                        data = json.loads(candidate)
                    except Exception as e2:
                        print("LLM raw content (failed JSON candidate):", repr(candidate))
                        raise ValueError(f"LLM did not return valid JSON: {e2}")
                else:
                    print("LLM raw content (failed JSON, no braces found):", repr(clean))
                    raise ValueError(
                        "LLM did not return valid JSON: no JSON object could be extracted."
                    )

        fmea_json = data.get("fmea", [])
        ppr_json = data.get("ppr", {"products": [], "processes": [], "resources": []})
        return fmea_json, ppr_json


        def generate_ppr_from_text(
            self,
            context_text: str,
            ppr_hint: dict | None = None,
        ) -> dict:
            """
            PPR-only generation from description + optional context.
            Expects to return JSON with keys:
            input_products, products, processes, resources.
            """
            system = (
                "You are an expert in manufacturing PPR (Product-Process-Resource) "
                "for process FMEA contexts.\n"
                "Given a production description and optional context (such as sample FMEA rows "
                "or file metadata), return ONE JSON object with exactly these keys:\n"
                "  - input_products: list of consumables / base materials fed into the process\n"
                "  - products: list of output products or workpieces\n"
                "  - processes: list of manufacturing or inspection processes\n"
                "  - resources: list of equipment, tools, fixtures, robots, sensors, etc.\n"
                "Use short, deduplicated strings. Do not include explanations or any other keys."
            )
    
            hint_json = json.dumps(ppr_hint or {}, ensure_ascii=False)
            user = (
                "Production description and context (may be JSON):\n"
                f"{context_text}\n\n"
                "Existing PPR hints (may be empty):\n"
                f"{hint_json}\n\n"
                "Return only one JSON object with keys: "
                "input_products, products, processes, resources."
            )
    
            content = self._chat(system, user, temperature=0.2, max_tokens=1600)
    
            if not content or not str(content).strip():
                raise ValueError("LLM did not return any content (empty response).")
    
            # Robust JSON parsing (same pattern as generate_fmea_and_ppr_json)
            try:
                data = json.loads(content)
            except Exception:
                clean = _sanitize_common(content)
                try:
                    data = json.loads(clean)
                except Exception:
                    text = clean.strip()
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = text[start : end + 1]
                        data = json.loads(candidate)
                    else:
                        raise ValueError(
                            "LLM did not return valid JSON for PPR-only generation."
                        )
    
            if not isinstance(data, dict):
                raise ValueError("PPR-only LLM result is not a JSON object.")
    
            # Normalize and ensure the four keys exist
            ppr = {
                "input_products": data.get("input_products", []) or [],
                "products": data.get("products", []) or [],
                "processes": data.get("processes", []) or [],
                "resources": data.get("resources", []) or [],
            }
            return ppr

        def generate_fmea_rows_json(self, context_text: str, ppr_hint: dict) -> list:
            """
            Generate ONLY FMEA rows from the given context.
            The LLM must return a JSON ARRAY of row objects, no wrapper.
            PPR is completely ignored here.
            """
            system = (
                "You are an expert in manufacturing Process FMEA (DIN EN 60812/APIS).\n"
                "From the provided context, you must propose additional FMEA rows.\n"
                "Output ONLY a JSON ARRAY (no outer object) where each element is one FMEA row.\n"
                "Use EXACTLY these keys for every row:\n"
                "system_element,function,potential_failure,c1,"
                "potential_effect,s1,c2,c3,"
                "potential_cause,o1,current_preventive_action,"
                "current_detection_action,d1,rpn1,"
                "recommended_action,rd,action_taken,"
                "s2,o2,d2,rpn2,notes.\n"
                "Do not output any PPR or other keys."
            )
            user = (
                "Context (may be JSON with KB FMEA rows and a PPR description):\n"
                f"{context_text}\n\n"
                "Now return ONLY the JSON ARRAY of FMEA rows as specified."
            )
    
            content = self._chat(system, user, temperature=0.2, max_tokens=3200)
    
            if not content or not str(content).strip():
                return []
    
            # Parse as JSON array, with a small fallback
            try:
                data = json.loads(content)
            except Exception:
                clean = _sanitize_common(content)
                text = clean.strip()
                start = text.find("[")
                end = text.rfind("]")
                if start == -1 or end <= start:
                    return []
                candidate = text[start : end + 1]
                data = json.loads(candidate)
    
            return data if isinstance(data, list) else []


    def normalize_label(self, text: str) -> str:
        return " ".join((text or "").strip().split())

# Global instance used by app
llm = LLM(model_name="sonar-pro")

# ---------------------------
# Embeddings (unchanged)
# ---------------------------

class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # all-MiniLM-L6-v2 => 384 dimensions
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """
        Return a single embedding vector (list of floats) for the input text.
        """
        if not isinstance(text, str):
            text = str(text)
        vec = self.model.encode([text], normalize_embeddings=True)
        # vec.shape = (1, 384)
        return vec[0].astype(np.float32).tolist()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)
