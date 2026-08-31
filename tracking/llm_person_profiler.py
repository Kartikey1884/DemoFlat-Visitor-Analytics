from __future__ import annotations

import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from config import Config, get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Standard browser User-Agent to prevent Cloudflare Error 1010 WAF blocks
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Supported LLM Providers & default models (including latest free-tier models)
LLM_PROVIDERS_CONFIG = {
    "groq": {
        "name": "Groq (High-Speed Free Tier LLM & Vision)",
        "default_model": "allam-2-7b",
        "available_models": [
            "allam-2-7b",
            "groq/compound-mini",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "groq/compound",
        ],
        "default_endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "key_env_var": "GROQ_API_KEY",
        "key_prefix": "gsk_",
    },
    "gemini": {
        "name": "Google Gemini (Free AI Studio Tier)",
        "default_model": "gemini-1.5-flash",
        "available_models": [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.5-flash",
            "gemini-3.6-flash",
        ],
        "default_endpoint": "https://generativelanguage.googleapis.com/v1beta/models",
        "key_env_var": "GEMINI_API_KEY",
        "key_prefix": "AIza",
    },
    "local": {
        "name": "Built-in Local Vision Engine (100% Free & Offline)",
        "default_model": "color-spatial-clustering",
        "available_models": ["color-spatial-clustering"],
        "default_endpoint": "",
        "key_env_var": "",
        "key_prefix": "",
    },
    "ollama": {
        "name": "Ollama / Local Custom Vision (100% Free)",
        "default_model": "llama3.2-vision",
        "available_models": [
            "llama3.2-vision",
            "llava",
            "minicpm-v",
            "qwen2.5-coder",
        ],
        "default_endpoint": "http://localhost:11434/v1/chat/completions",
        "key_env_var": "OLLAMA_API_KEY",
        "key_prefix": "",
    },
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o-mini",
        "available_models": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
        "default_endpoint": "https://api.openai.com/v1/chat/completions",
        "key_env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
    },
    "claude": {
        "name": "Anthropic Claude",
        "default_model": "claude-3-5-sonnet-20241022",
        "available_models": [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ],
        "default_endpoint": "https://api.anthropic.com/v1/messages",
        "key_env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
    },
}


@dataclass
class PersonSemanticProfile:
    upper_clothing: str = "Unknown Top"
    upper_color: str = "Unknown"
    lower_clothing: str = "Unknown Bottom"
    lower_color: str = "Unknown"
    accessories: List[str] = field(default_factory=list)
    build_and_gender: str = "Person"
    persona_summary: str = "Person in frame"
    suggested_role: str = "visitor"
    role_confidence: float = 0.8
    role_reasoning: str = ""
    is_human: bool = True
    is_tv_or_animal: bool = False
    extracted_by: str = "vision"
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upper_clothing": self.upper_clothing,
            "upper_color": self.upper_color,
            "lower_clothing": self.lower_clothing,
            "lower_color": self.lower_color,
            "accessories": self.accessories,
            "build_and_gender": self.build_and_gender,
            "persona_summary": self.persona_summary,
            "suggested_role": self.suggested_role,
            "role_confidence": self.role_confidence,
            "role_reasoning": self.role_reasoning,
            "is_human": self.is_human,
            "is_tv_or_animal": self.is_tv_or_animal,
            "extracted_by": self.extracted_by,
            "extracted_at": self.extracted_at.isoformat() if self.extracted_at else None,
        }


@dataclass
class LLMReIDDecision:
    timestamp: datetime
    track_id: int
    decision: str  # "MATCH" or "NEW_PERSON"
    matched_global_id: Optional[str]
    confidence: float
    reasoning: str
    persona_summary: str
    engine: str

    def to_dict(self) -> dict:
        return {
            "time": self.timestamp.strftime("%H:%M:%S"),
            "track_id": self.track_id,
            "decision": "🔁 Returning Match" if self.decision == "MATCH" else "🆕 New Visitor",
            "matched_id": self.matched_global_id or "New Identity",
            "confidence": f"{int(self.confidence * 100)}%",
            "reasoning": self.reasoning,
            "persona": self.persona_summary,
            "engine": self.engine,
        }


class LLMPersonProfiler:
    """
    Asynchronous LLM Visual Persona Profiler & Re-ID Arbitrator.
    Supports Groq, Gemini, OpenAI, Claude, Ollama, and Local Fallback.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.llm_cfg = self.config.llm
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm_profiler")
        self._cache: Dict[str, PersonSemanticProfile] = {}
        self._llm_decisions: List[LLMReIDDecision] = []
        self._rate_limited_until: Dict[str, float] = {}
        self._executor_lock = threading.Lock()
        self._pending_tasks: set[str] = set()

    def fetch_available_models(self, provider: str, api_key: str, base_url: str = "") -> List[str]:
        """Queries the provider's API to get all currently active models."""
        provider = provider.lower().strip()
        try:
            if provider == "groq" and api_key:
                url = "https://api.groq.com/openai/v1/models"
                headers = {**HTTP_HEADERS, "Authorization": f"Bearer {api_key}"}
                resp = requests.get(url, headers=headers, timeout=6.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", []) if "whisper" not in m["id"].lower()]
                    return sorted(models)
            elif provider == "gemini" and api_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
                resp = requests.get(url, headers=HTTP_HEADERS, timeout=6.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m["name"].replace("models/", "")
                        for m in data.get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                        and not m["name"].startswith("models/embedding")
                    ]
                    return sorted(models)
            elif provider == "openai" and api_key:
                url = "https://api.openai.com/v1/models"
                headers = {**HTTP_HEADERS, "Authorization": f"Bearer {api_key}"}
                resp = requests.get(url, headers=headers, timeout=6.0)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", []) if "gpt" in m["id"].lower()]
                    return sorted(models)
            elif provider == "ollama":
                endpoint = (base_url or "http://localhost:11434").replace("/v1/chat/completions", "").rstrip("/")
                url = f"{endpoint}/api/tags"
                resp = requests.get(url, headers=HTTP_HEADERS, timeout=4.0)
                if resp.status_code == 200:
                    data = resp.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.debug("Failed to fetch models for %s: %s", provider, exc)

        return LLM_PROVIDERS_CONFIG.get(provider, {}).get("available_models", [])

    def profile_person_async(
        self,
        crop: np.ndarray,
        global_person: Any,
        callback: Optional[callable] = None,
    ) -> None:
        """Asynchronously extracts LLM semantic profile in a background worker thread."""
        gid = getattr(global_person, "global_id", "P-000")
        if gid in self._pending_tasks or crop is None or crop.size == 0:
            return

        with self._executor_lock:
            self._pending_tasks.add(gid)

        crop_copy = crop.copy()

        def _worker():
            try:
                profile = self.profile_person(crop_copy, gid)
                if hasattr(global_person, "semantic_profile"):
                    global_person.semantic_profile = profile
                if callback:
                    callback(global_person, profile)
                logger.info("LLM Semantic Profile generated for %s via %s: %s", gid, profile.extracted_by, profile.persona_summary)
            except Exception as exc:
                logger.warning("LLM Profiling error for %s: %s", gid, exc)
            finally:
                with self._executor_lock:
                    self._pending_tasks.discard(gid)

        t = threading.Thread(target=_worker, daemon=True, name=f"LLM-Profiler-{gid}")
        t.start()

    def profile_person(self, crop: np.ndarray, global_id: str = "") -> PersonSemanticProfile:
        """Extracts structured semantic persona profile using the configured LLM provider."""
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return PersonSemanticProfile()

        llm_cfg = self.config.llm
        provider = (llm_cfg.provider or "groq").lower().strip()
        model_name = llm_cfg.model_name
        api_key = llm_cfg.api_key or os.environ.get(LLM_PROVIDERS_CONFIG.get(provider, {}).get("key_env_var", "")) or os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        base_url = llm_cfg.base_url

        if provider == "groq" and api_key:
            res = self._call_groq_vision(crop, api_key, model_name or "llama-3.3-70b-versatile", global_id)
            if res is not None:
                return res

        elif provider == "gemini" and api_key:
            res = self._call_gemini_vision(crop, api_key, model_name or "gemini-1.5-flash", global_id)
            if res is not None:
                return res

        elif provider == "openai" and api_key:
            res = self._call_openai_vision(crop, api_key, model_name or "gpt-4o-mini", global_id)
            if res is not None:
                return res

        elif provider == "claude" and api_key:
            res = self._call_claude_vision(crop, api_key, model_name or "claude-3-5-sonnet-20241022", global_id)
            if res is not None:
                return res

        elif provider == "ollama" or (base_url and "http" in base_url):
            res = self._call_ollama_custom_vision(crop, base_url or "http://localhost:11434/v1/chat/completions", model_name or "llama3.2-vision", api_key, global_id)
            if res is not None:
                return res

        return self._extract_local_vision_profile(crop, global_id)

    def decide_reid_match(
        self,
        crop: np.ndarray,
        track_id: int,
        current_profile: PersonSemanticProfile,
        candidates: List[Tuple[Any, float]],  # List of (GlobalPerson, visual_similarity)
    ) -> LLMReIDDecision:
        """
        Uses the LLM to arbitrate whether a newly detected person is a returning visitor or a new person.
        """
        now = datetime.now()
        llm_cfg = self.config.llm
        provider = (llm_cfg.provider or "groq").lower().strip()
        model_name = llm_cfg.model_name
        api_key = llm_cfg.api_key or os.environ.get(LLM_PROVIDERS_CONFIG.get(provider, {}).get("key_env_var", "")) or os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

        # If no candidates, it's a new visitor
        if not candidates:
            reason = f"First-time entry detected in frame. Extracted visual persona: {current_profile.persona_summary}."
            return LLMReIDDecision(
                timestamp=now,
                track_id=track_id,
                decision="NEW_PERSON",
                matched_global_id=None,
                confidence=1.0,
                reasoning=reason,
                persona_summary=current_profile.persona_summary,
                engine=f"{provider}:{model_name}" if api_key else "local_vision",
            )

        # Candidate descriptions
        cand_descriptions = []
        for p, sim in candidates:
            prof = getattr(p, "semantic_profile", None)
            prof_str = prof.persona_summary if prof else f"Top: {prof.upper_clothing if prof else 'Unknown'}, Bottom: {prof.lower_clothing if prof else 'Unknown'}"
            cand_descriptions.append(
                f"- ID: {p.global_id} ({p.display_name}) | Stored Visual Profile: \"{prof_str}\" | Feature Match Score: {sim:.2f}"
            )

        candidates_text = "\n".join(cand_descriptions)

        # Call LLM reasoning if API key available
        if provider == "groq" and api_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {**HTTP_HEADERS, "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                prompt = (
                    "You are the AI CCTV Biometric Re-Identification Arbiter.\n"
                    f"A person just appeared on camera (Track ID: {track_id}).\n"
                    f"Their visual appearance is:\n"
                    f"• Upper clothing: {current_profile.upper_clothing}\n"
                    f"• Lower clothing: {current_profile.lower_clothing}\n"
                    f"• Accessories: {', '.join(current_profile.accessories) if current_profile.accessories else 'None'}\n"
                    f"• Summary: {current_profile.persona_summary}\n\n"
                    f"Compare against these registered visitors who previously left into rooms:\n"
                    f"{candidates_text}\n\n"
                    "Determine whether this newly appeared person is RE-IDENTIFIED as one of the existing visitors (e.g. returning from an interior room), or is a NEW UNIQUE VISITOR.\n"
                    "Output ONLY valid JSON:\n"
                    "{\n"
                    '  "decision": "MATCH" or "NEW_PERSON",\n'
                    '  "matched_global_id": "P-001" or null,\n'
                    '  "confidence": 0.92,\n'
                    '  "reasoning": "Clear concise reason comparing upper/lower clothing and accessories."\n'
                    "}\n"
                    "Return raw JSON only."
                )
                payload = {
                    "model": model_name or "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 250,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=5.0)
                if resp.status_code == 200:
                    data = json.loads(resp.json()["choices"][0]["message"]["content"].replace("```json", "").replace("```", "").strip())
                    dec = data.get("decision", "MATCH")
                    matched_id = data.get("matched_global_id")
                    conf = float(data.get("confidence", 0.90))
                    reason = data.get("reasoning", "Matched via Groq AI visual arbitration.")
                    return LLMReIDDecision(
                        timestamp=now,
                        track_id=track_id,
                        decision=dec,
                        matched_global_id=matched_id,
                        confidence=conf,
                        reasoning=reason,
                        persona_summary=current_profile.persona_summary,
                        engine=f"groq:{model_name}",
                    )
            except Exception as exc:
                logger.debug("Groq LLM Re-ID arbitration fallback: %s", exc)

        # High-Accuracy Deterministic Fallback
        best_cand, best_sim = candidates[0]
        if best_sim >= self.config.reid.similarity_threshold:
            return LLMReIDDecision(
                timestamp=now,
                track_id=track_id,
                decision="MATCH",
                matched_global_id=best_cand.global_id,
                confidence=best_sim,
                reasoning=f"Matched returning visitor {best_cand.global_id} ({int(best_sim*100)}% visual similarity on clothing & appearance memory).",
                persona_summary=current_profile.persona_summary,
                engine="hybrid_vision",
            )
        else:
            return LLMReIDDecision(
                timestamp=now,
                track_id=track_id,
                decision="NEW_PERSON",
                matched_global_id=None,
                confidence=1.0 - best_sim,
                reasoning=f"Distinct clothing/appearance from all registered visitors (max similarity {int(best_sim*100)}% < threshold).",
                persona_summary=current_profile.persona_summary,
                engine="hybrid_vision",
            )

    def test_connection(
        self,
        provider: str,
        api_key: str,
        model_name: str = "",
        base_url: str = "",
    ) -> Tuple[bool, str]:
        """Tests the LLM API configuration with a test crop."""
        provider = provider.lower().strip()
        test_crop = np.zeros((100, 60, 3), dtype=np.uint8)
        cv2.rectangle(test_crop, (5, 10), (55, 50), (220, 140, 60), -1)
        cv2.rectangle(test_crop, (5, 50), (55, 90), (35, 35, 35), -1)

        try:
            if provider == "groq":
                if not api_key:
                    return False, "Groq API key is missing. Please enter your gsk_... key."
                m = model_name or "llama-3.3-70b-versatile"
                try:
                    p = self._call_groq_vision(test_crop, api_key, m, "TEST", raise_on_error=True)
                    if p:
                        return True, f"Connected to Groq ({p.extracted_by}): {p.persona_summary}"
                    return False, f"Groq API returned an empty response for {m}."
                except Exception as exc:
                    return False, str(exc)

            elif provider == "gemini":
                if not api_key:
                    return False, "Gemini API key is missing. Please enter your AIza... key."
                m = model_name or "gemini-1.5-flash"
                try:
                    p = self._call_gemini_vision(test_crop, api_key, m, "TEST", raise_on_error=True)
                    if p:
                        return True, f"Connected to Google Gemini ({p.extracted_by}): {p.persona_summary}"
                    return False, "Gemini API failed."
                except Exception as exc:
                    return False, str(exc)

            elif provider == "openai":
                if not api_key:
                    return False, "OpenAI API key is missing. Please enter your sk-... key."
                m = model_name or "gpt-4o-mini"
                try:
                    p = self._call_openai_vision(test_crop, api_key, m, "TEST", raise_on_error=True)
                    if p:
                        return True, f"Connected to OpenAI ({p.extracted_by}): {p.persona_summary}"
                    return False, "OpenAI API call failed."
                except Exception as exc:
                    return False, str(exc)

            elif provider == "claude":
                if not api_key:
                    return False, "Anthropic API key is missing."
                m = model_name or "claude-3-5-sonnet-20241022"
                try:
                    p = self._call_claude_vision(test_crop, api_key, m, "TEST", raise_on_error=True)
                    if p:
                        return True, f"Connected to Claude ({p.extracted_by}): {p.persona_summary}"
                    return False, "Claude API call failed."
                except Exception as exc:
                    return False, str(exc)

            elif provider == "ollama":
                endpoint = base_url or "http://localhost:11434/v1/chat/completions"
                p = self._call_ollama_custom_vision(test_crop, endpoint, model_name or "llama3.2-vision", api_key, "TEST", raise_on_error=True)
                if p:
                    return True, f"Connected to Ollama/Custom endpoint: {p.persona_summary}"
                return False, f"Could not reach Ollama at {endpoint}."

            elif provider == "local":
                p = self._extract_local_vision_profile(test_crop, "TEST")
                return True, f"Local Vision Engine Ready: {p.persona_summary}"

            return False, f"Unknown provider: {provider}"
        except Exception as exc:
            return False, f"Connection failed: {str(exc)}"

    def _call_groq_vision(
        self,
        crop: np.ndarray,
        api_key: str,
        model_name: str,
        global_id: str,
        raise_on_error: bool = False,
    ) -> Optional[PersonSemanticProfile]:
        """Calls Groq API with native Vision or Hybrid Visual Attribute Reasoning."""
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf).decode("utf-8")

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                **HTTP_HEADERS,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            model = model_name or "llama-3.3-70b-versatile"
            is_explicit_vision_model = any(v in model.lower() for v in ["vision", "vl", "scout", "llava"])

            # 1. If it's a multimodal vision model, send image_url directly
            if is_explicit_vision_model:
                prompt = (
                    "You are an expert CCTV visual profiler. Analyze this person image crop and return ONLY a valid JSON object with: "
                    "upper_clothing (e.g. Light blue shirt), upper_color (e.g. Light Blue), lower_clothing (e.g. Dark navy trousers), "
                    "lower_color (e.g. Dark Grey), accessories (list of items like Shoes Covers, Mask, Watch, Glasses), "
                    "build_and_gender (e.g. Adult Male), persona_summary (one sentence distinct visual signature). "
                    "Return raw JSON only."
                )
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }],
                    "temperature": 0.1,
                    "max_tokens": 300,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
                if resp.status_code == 200:
                    result = resp.json()
                    text = result["choices"][0]["message"]["content"]
                    clean_text = text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean_text)
                    return PersonSemanticProfile(
                        upper_clothing=data.get("upper_clothing", "Unknown Top"),
                        upper_color=data.get("upper_color", "Unknown"),
                        lower_clothing=data.get("lower_clothing", "Unknown Bottom"),
                        lower_color=data.get("lower_color", "Unknown"),
                        accessories=data.get("accessories", []),
                        build_and_gender=data.get("build_and_gender", "Adult Person"),
                        persona_summary=data.get("persona_summary", f"Person {global_id}"),
                        extracted_by=f"groq-vision:{model}",
                        extracted_at=datetime.now(),
                    )
                logger.info("Groq vision call returned %d for %s. Using Groq hybrid reasoning.", resp.status_code, model)

            # 2. Hybrid Visual Reasoning (works for 120b, compound, allam, llama, etc.)
            local_prof = self._extract_local_vision_profile(crop, global_id)
            reasoning_prompt = (
                f"You are an AI CCTV Biometric Profiler for real estate flat visits.\n"
                f"A visual feature detector extracted these attributes from a bounding box crop:\n"
                f"- Upper clothing: {local_prof.upper_clothing} (dominant color: {local_prof.upper_color})\n"
                f"- Lower clothing: {local_prof.lower_clothing} (dominant color: {local_prof.lower_color})\n"
                f"- Accessories: {', '.join(local_prof.accessories) if local_prof.accessories else 'None'}\n"
                f"- Build: {local_prof.build_and_gender}\n"
                f"- Initial Assessment: {'Real Human' if local_prof.is_human else 'Non-human / Animal / TV Screen'}\n\n"
                f"Analyze if this is a REAL HUMAN BEING (visitor/staff) or an ANIMAL (e.g. horse, dog, wildlife on a TV screen/poster) / inanimate object.\n"
                f"Output ONLY a valid JSON object:\n"
                "{\n"
                f'  "is_human": {json.dumps(local_prof.is_human)},\n'
                f'  "is_tv_or_animal": {json.dumps(local_prof.is_tv_or_animal)},\n'
                f'  "upper_clothing": "{local_prof.upper_clothing}",\n'
                f'  "upper_color": "{local_prof.upper_color}",\n'
                f'  "lower_clothing": "{local_prof.lower_clothing}",\n'
                f'  "lower_color": "{local_prof.lower_color}",\n'
                f'  "accessories": {json.dumps(local_prof.accessories)},\n'
                f'  "build_and_gender": "{local_prof.build_and_gender}",\n'
                '  "persona_summary": "One concise sentence describing the person (or indicating non-human/TV screen if applicable)"\n'
                "}\n"
                "Return raw JSON only."
            )

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": reasoning_prompt}],
                "temperature": 0.1,
                "max_tokens": 300,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code != 200:
                err_text = resp.text
                try:
                    err_json = resp.json()
                    msg = err_json.get("error", {}).get("message", err_text)
                except Exception:
                    msg = err_text
                if raise_on_error:
                    raise ValueError(f"Groq API HTTP {resp.status_code}: {msg}")
                logger.warning("Groq API Error %d: %s", resp.status_code, msg)
                return None

            result = resp.json()
            text = result["choices"][0]["message"]["content"]
            clean_text = text.strip()
            if "```" in clean_text:
                parts = clean_text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        clean_text = part.replace("json", "").strip()
                        break
            if "{" in clean_text and "}" in clean_text:
                s_idx = clean_text.find("{")
                e_idx = clean_text.rfind("}") + 1
                clean_text = clean_text[s_idx:e_idx]
            try:
                data = json.loads(clean_text)
            except Exception:
                data = {}

            is_human = data.get("is_human", local_prof.is_human)
            is_tv_or_animal = data.get("is_tv_or_animal", local_prof.is_tv_or_animal)

            return PersonSemanticProfile(
                upper_clothing=data.get("upper_clothing", local_prof.upper_clothing),
                upper_color=data.get("upper_color", local_prof.upper_color),
                lower_clothing=data.get("lower_clothing", local_prof.lower_clothing),
                lower_color=data.get("lower_color", local_prof.lower_color),
                accessories=data.get("accessories", local_prof.accessories),
                build_and_gender=data.get("build_and_gender", local_prof.build_and_gender),
                persona_summary=data.get("persona_summary", local_prof.persona_summary),
                is_human=is_human,
                is_tv_or_animal=is_tv_or_animal,
                suggested_role="visitor" if is_human and not is_tv_or_animal else "non_human",
                extracted_by=f"groq:{model}",
                extracted_at=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Groq call failed: %s", exc)
            if raise_on_error:
                raise exc
            return None

    def _call_gemini_vision(
        self,
        crop: np.ndarray,
        api_key: str,
        model_name: str,
        global_id: str,
        raise_on_error: bool = False,
    ) -> Optional[PersonSemanticProfile]:
        """Calls Google Gemini Vision API."""
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf).decode("utf-8")

            model = (model_name or "gemini-2.5-flash").replace("models/", "").strip()
            headers = {**HTTP_HEADERS, "Content-Type": "application/json"}

            prompt = (
                "You are an expert CCTV biometric visual profiler. "
                "Analyze this person image crop and return ONLY a valid JSON object describing their clothing and appearance attributes:\n"
                "{\n"
                '  "upper_clothing": "concise description of top (e.g. Light blue formal dress shirt)",\n'
                '  "upper_color": "main top color (e.g. Light Blue)",\n'
                '  "lower_clothing": "concise description of pants (e.g. Dark navy formal trousers)",\n'
                '  "lower_color": "main bottom color (e.g. Dark Navy)",\n'
                '  "accessories": ["list of items like Glasses, Mask, Watch, Shoe Covers, Bag, Hat"],\n'
                '  "build_and_gender": "e.g. Adult Male, Slim athletic build",\n'
                '  "persona_summary": "One sentence distinct visual signature (e.g. Male in light blue formal shirt and dark trousers with blue shoe covers)"\n'
                "}\n"
                "Return raw JSON only without markdown fences."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}}
                    ]
                }],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 300}
            }

            # Try v1beta then v1 endpoint
            urls_to_try = [
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}",
            ]

            resp = None
            last_err = ""
            for url in urls_to_try:
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
                    if resp.status_code == 200:
                        break
                    else:
                        last_err = resp.text
                except Exception as req_err:
                    last_err = str(req_err)

            if resp is None or resp.status_code != 200:
                if raise_on_error:
                    msg = last_err
                    try:
                        err_json = json.loads(last_err)
                        msg = err_json.get("error", {}).get("message", last_err)
                    except Exception:
                        pass
                    raise ValueError(f"Gemini API HTTP {resp.status_code if resp else 500}: {msg}")
                return None

            result = resp.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)

            return PersonSemanticProfile(
                upper_clothing=data.get("upper_clothing", "Unknown Top"),
                upper_color=data.get("upper_color", "Unknown"),
                lower_clothing=data.get("lower_clothing", "Unknown Bottom"),
                lower_color=data.get("lower_color", "Unknown"),
                accessories=data.get("accessories", []),
                build_and_gender=data.get("build_and_gender", "Adult Person"),
                persona_summary=data.get("persona_summary", f"Person {global_id}"),
                extracted_by=f"gemini:{model}",
                extracted_at=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Gemini VLM API call failed: %s", exc)
            if raise_on_error:
                raise exc
            return None

    def _call_openai_vision(
        self,
        crop: np.ndarray,
        api_key: str,
        model_name: str,
        global_id: str,
        raise_on_error: bool = False,
    ) -> Optional[PersonSemanticProfile]:
        """Calls OpenAI GPT-4o Vision API."""
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf).decode("utf-8")

            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                **HTTP_HEADERS,
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            payload = {
                "model": model_name or "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this person crop and return a JSON object with: upper_clothing, upper_color, lower_clothing, lower_color, accessories (list), build_and_gender, persona_summary. Return raw JSON only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                "response_format": {"type": "json_object"},
                "max_tokens": 250,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code != 200:
                if raise_on_error:
                    raise ValueError(f"OpenAI API HTTP {resp.status_code}: {resp.text}")
                return None

            result = resp.json()
            text = result["choices"][0]["message"]["content"]
            data = json.loads(text)

            return PersonSemanticProfile(
                upper_clothing=data.get("upper_clothing", "Unknown Top"),
                upper_color=data.get("upper_color", "Unknown"),
                lower_clothing=data.get("lower_clothing", "Unknown Bottom"),
                lower_color=data.get("lower_color", "Unknown"),
                accessories=data.get("accessories", []),
                build_and_gender=data.get("build_and_gender", "Adult Person"),
                persona_summary=data.get("persona_summary", f"Person {global_id}"),
                extracted_by=f"openai:{model_name}",
                extracted_at=datetime.now(),
            )
        except Exception as exc:
            logger.debug("OpenAI VLM API call failed: %s", exc)
            if raise_on_error:
                raise exc
            return None

    def _call_claude_vision(
        self,
        crop: np.ndarray,
        api_key: str,
        model_name: str,
        global_id: str,
        raise_on_error: bool = False,
    ) -> Optional[PersonSemanticProfile]:
        """Calls Anthropic Claude Vision API."""
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf).decode("utf-8")

            url = "https://api.anthropic.com/v1/messages"
            headers = {
                **HTTP_HEADERS,
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            }

            prompt = (
                "Analyze this person crop and return ONLY a JSON object with: "
                "upper_clothing, upper_color, lower_clothing, lower_color, accessories (list), build_and_gender, persona_summary. "
                "Raw JSON only."
            )

            payload = {
                "model": model_name or "claude-3-5-sonnet-20241022",
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_img}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=8.0)
            if resp.status_code != 200:
                if raise_on_error:
                    raise ValueError(f"Claude API HTTP {resp.status_code}: {resp.text}")
                return None

            result = resp.json()
            text = result["content"][0]["text"]
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)

            return PersonSemanticProfile(
                upper_clothing=data.get("upper_clothing", "Unknown Top"),
                upper_color=data.get("upper_color", "Unknown"),
                lower_clothing=data.get("lower_clothing", "Unknown Bottom"),
                lower_color=data.get("lower_color", "Unknown"),
                accessories=data.get("accessories", []),
                build_and_gender=data.get("build_and_gender", "Adult Person"),
                persona_summary=data.get("persona_summary", f"Person {global_id}"),
                extracted_by=f"claude:{model_name}",
                extracted_at=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Claude VLM API call failed: %s", exc)
            if raise_on_error:
                raise exc
            return None

    def _call_ollama_custom_vision(
        self,
        crop: np.ndarray,
        endpoint: str,
        model_name: str,
        api_key: str,
        global_id: str,
        raise_on_error: bool = False,
    ) -> Optional[PersonSemanticProfile]:
        """Calls Ollama or Custom OpenAI-compatible Vision endpoint."""
        try:
            _, buf = cv2.imencode(".jpg", crop)
            b64_img = base64.b64encode(buf).decode("utf-8")

            headers = {**HTTP_HEADERS, "Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            prompt = "Analyze this person crop and return ONLY a JSON object with: upper_clothing, upper_color, lower_clothing, lower_color, accessories (list), build_and_gender, persona_summary."

            payload = {
                "model": model_name or "llama3.2-vision",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }],
                "temperature": 0.1,
            }

            resp = requests.post(endpoint, headers=headers, json=payload, timeout=10.0)
            if resp.status_code != 200:
                if raise_on_error:
                    raise ValueError(f"Ollama/Custom API HTTP {resp.status_code}: {resp.text}")
                return None

            result = resp.json()
            text = result["choices"][0]["message"]["content"]
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)

            return PersonSemanticProfile(
                upper_clothing=data.get("upper_clothing", "Unknown Top"),
                upper_color=data.get("upper_color", "Unknown"),
                lower_clothing=data.get("lower_clothing", "Unknown Bottom"),
                lower_color=data.get("lower_color", "Unknown"),
                accessories=data.get("accessories", []),
                build_and_gender=data.get("build_and_gender", "Adult Person"),
                persona_summary=data.get("persona_summary", f"Person {global_id}"),
                extracted_by=f"ollama:{model_name}",
                extracted_at=datetime.now(),
            )
        except Exception as exc:
            logger.debug("Ollama/Custom Vision API call failed: %s", exc)
            if raise_on_error:
                raise exc
            return None

    def _extract_local_vision_profile(self, crop: np.ndarray, global_id: str) -> PersonSemanticProfile:
        """Deterministic local vision attribute extractor for offline semantic descriptions."""
        if crop is None or crop.size == 0:
            return PersonSemanticProfile()
        h, w = crop.shape[:2]
        aspect_ratio = h / max(w, 1)
        is_partial_head = (aspect_ratio < 1.35 or h < 85)

        # Detect non-human artifacts (animals on TV, wide objects, posters)
        is_non_human = (aspect_ratio < 0.90) or (h < 40 and w < 30)

        if is_non_human:
            is_human = False
            is_tv_or_animal = True
            upper_desc = "Non-human object"
            lower_desc = "Non-human object"
            upper_color = "Unknown"
            lower_color = "Unknown"
            accessories = []
            build_str = "Non-human"
            suggested_role = "non_human"
            role_conf = 0.90
            role_reason = "Non-human aspect ratio / TV screen artifact"
            summary = "Non-human object / TV screen video detected"
        elif is_partial_head:
            is_human = True
            is_tv_or_animal = False
            # Crop is predominantly head/shoulders
            upper_crop = crop[int(h * 0.35) :, int(w * 0.10) : int(w * 0.90)]
            upper_color = self._get_dominant_color_name(upper_crop)
            lower_color = "Occluded / Partial View"
            feet_color = "None"
            accessories = []
            build_str = "Seated / Partial view"
            upper_desc = f"{upper_color} Top/Shirt"
            lower_desc = "Lower body occluded"
            suggested_role = "visitor"
            role_conf = 0.70
            role_reason = "Partial upper body / head view"
            summary = f"Person in {upper_color} top (Head & upper body visible)"
        else:
            is_human = True
            is_tv_or_animal = False
            upper_crop = crop[int(h * 0.15) : int(h * 0.50), int(w * 0.15) : int(w * 0.85)]
            lower_crop = crop[int(h * 0.50) : int(h * 0.90), int(w * 0.15) : int(w * 0.85)]
            feet_crop = crop[int(h * 0.88) :, int(w * 0.15) : int(w * 0.85)]

            upper_color = self._get_dominant_color_name(upper_crop)
            lower_color = self._get_dominant_color_name(lower_crop)
            feet_color = self._get_dominant_color_name(feet_crop)

            accessories = []
            if "Blue" in feet_color or "Light Blue" in feet_color:
                accessories.append("Blue Shoe Covers")

            head_crop = crop[0 : int(h * 0.20), :]
            if head_crop.size > 0:
                hsv_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                face_lower = hsv_head[int(head_crop.shape[0] * 0.5) :, :]
                if face_lower.size > 0:
                    mean_s = np.mean(face_lower[:, :, 1])
                    mean_v = np.mean(face_lower[:, :, 2])
                    if mean_v > 180 and mean_s < 80:
                        accessories.append("Face Mask")

            build_str = "Tall build" if aspect_ratio > 2.8 else "Medium athletic build"

            # Attire Role Classification (Sales Agent Formal Dress vs Casual Visitor)
            is_formal_top = upper_color in ["White", "Light Blue", "Navy Blue", "Black / Dark", "Grey"]
            is_formal_bottom = lower_color in ["Black / Dark", "Grey", "Navy Blue", "Neutral Dark"]
            
            if is_formal_top and is_formal_bottom:
                suggested_role = "sales_person"
                role_conf = 0.82
                role_reason = f"Formal business attire detected ({upper_color} collared top + {lower_color} trousers)"
            else:
                suggested_role = "visitor"
                role_conf = 0.88
                role_reason = f"Casual attire ({upper_color} top + {lower_color} bottom)"

            upper_desc = f"{upper_color} Top/Shirt"
            lower_desc = f"{lower_color} Pants/Trousers"
            acc_str = f" with {', '.join(accessories)}" if accessories else ""
            role_label = " [Sales Agent]" if suggested_role == "sales_person" else ""
            summary = f"Person in {upper_color} shirt and {lower_color} trousers{acc_str} ({build_str}){role_label}"

        return PersonSemanticProfile(
            upper_clothing=upper_desc,
            upper_color=upper_color,
            lower_clothing=lower_desc,
            lower_color=lower_color,
            accessories=accessories,
            build_and_gender=build_str,
            persona_summary=summary,
            suggested_role=suggested_role,
            role_confidence=role_conf,
            role_reasoning=role_reason,
            is_human=is_human,
            is_tv_or_animal=is_tv_or_animal,
            extracted_by="local_vision",
            extracted_at=datetime.now(),
        )

    def _get_dominant_color_name(self, region: np.ndarray) -> str:
        """Maps an image region to human-understandable color names."""
        if region is None or region.size == 0:
            return "Dark"

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mean_h = np.mean(h)
        mean_s = np.mean(s)
        mean_v = np.mean(v)

        if mean_v < 45:
            return "Black / Dark"
        if mean_v > 200 and mean_s < 40:
            return "White"
        if mean_s < 45:
            return "Grey"

        if 95 <= mean_h <= 130:
            return "Light Blue" if mean_v > 160 else "Navy Blue"
        if 80 <= mean_h < 95:
            return "Cyan / Teal"
        if 35 <= mean_h < 80:
            return "Green"
        if 18 <= mean_h < 35:
            return "Khaki / Yellow" if mean_s < 140 else "Yellow"
        if 8 <= mean_h < 18:
            return "Brown" if mean_v < 130 else "Orange"
        if mean_h < 8 or mean_h >= 170:
            return "Dark Red / Maroon" if mean_v < 120 else "Red / Pink"
        if 130 < mean_h < 170:
            return "Purple / Violet"

        return "Neutral Dark"

    @staticmethod
    def compute_semantic_similarity(
        prof1: PersonSemanticProfile,
        prof2: PersonSemanticProfile,
    ) -> float:
        """Computes semantic attribute similarity between two person profiles."""
        score = 0.0
        if prof1.upper_color == prof2.upper_color and prof1.upper_color != "Unknown":
            score += 0.40
        elif any(c in prof2.upper_color for c in prof1.upper_color.split()):
            score += 0.25

        if prof1.lower_color == prof2.lower_color and prof1.lower_color != "Unknown":
            score += 0.40
        elif any(c in prof2.lower_color for c in prof1.lower_color.split()):
            score += 0.25

        common_acc = set(prof1.accessories).intersection(set(prof2.accessories))
        if common_acc:
            score += 0.20
        elif not prof1.accessories and not prof2.accessories:
            score += 0.10

        return max(0.0, min(1.0, score))
