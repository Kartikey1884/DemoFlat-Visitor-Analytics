from __future__ import annotations

import os
import streamlit as st
from typing import Any, Optional

from config import Config, get_config
from tracking.llm_person_profiler import LLMPersonProfiler, LLM_PROVIDERS_CONFIG


PROVIDER_DISPLAY_MAP = {
    "Groq": "groq",
    "Google Gemini": "gemini",
    "OpenAI": "openai",
    "Anthropic Claude": "claude",
    "Ollama / Custom API": "ollama",
    "Built-in Local Vision": "local",
}

REVERSE_PROVIDER_MAP = {v: k for k, v in PROVIDER_DISPLAY_MAP.items()}


def render_llm_config_card(cfg: Optional[Config] = None, db: Optional[Any] = None, expanded: bool = False) -> None:
    """
    Renders an interactive configuration card to select LLM provider, enter API key,
    choose model, test connection, and apply settings before video processing.
    """
    cfg = cfg or get_config()
    llm_cfg = cfg.llm

    current_provider_key = (llm_cfg.provider or "groq").lower().strip()
    current_provider_disp = REVERSE_PROVIDER_MAP.get(current_provider_key, "Groq")

    with st.expander("🤖 AI Vision LLM & Persona Profiler Settings", expanded=expanded):
        st.markdown(
            "Select your **LLM Provider** (Groq, Gemini, OpenAI, Claude, Ollama) "
            "and enter your API key to enable visual persona profiling & semantic re-identification."
        )

        c_prov, c_model = st.columns([1, 1])
        with c_prov:
            prov_options = list(PROVIDER_DISPLAY_MAP.keys())
            def_idx = prov_options.index(current_provider_disp) if current_provider_disp in prov_options else 0
            selected_disp = st.selectbox(
                "LLM Provider",
                prov_options,
                index=def_idx,
                key="ui_llm_provider_select",
                help="Groq offers ultra-fast sub-second inference. Local vision works 100% offline.",
            )
            provider_key = PROVIDER_DISPLAY_MAP[selected_disp]

        prov_info = LLM_PROVIDERS_CONFIG.get(provider_key, {})
        cached_models_key = f"cached_models_{provider_key}"
        avail_models = st.session_state.get(cached_models_key, prov_info.get("available_models", ["default"]))
        curr_model = llm_cfg.model_name or prov_info.get("default_model", "")

        with c_model:
            if provider_key == "local":
                st.text_input("Vision Engine", value="Built-in Color-Spatial Clustering", disabled=True)
                model_name = "color-spatial-clustering"
            else:
                model_idx = avail_models.index(curr_model) if curr_model in avail_models else 0
                chosen_model = st.selectbox(
                    "Model Name",
                    avail_models + ["Custom Model Name..."],
                    index=model_idx,
                    key=f"ui_llm_model_{provider_key}",
                )
                if chosen_model == "Custom Model Name...":
                    model_name = st.text_input("Enter Custom Model Name", value=curr_model, key="ui_llm_custom_model")
                else:
                    model_name = chosen_model

        # API Key & Base URL
        col_key, col_url = st.columns([2, 1] if provider_key == "ollama" else [3, 1])

        with col_key:
            if provider_key == "local":
                st.info("💡 Local Vision Engine does not require any API keys and runs 100% offline.")
                api_key = ""
            else:
                env_key_name = prov_info.get("key_env_var", "")
                existing_key = llm_cfg.api_key or os.environ.get(env_key_name, "")
                placeholder = f"e.g. {prov_info.get('key_prefix', '')}..."
                api_key = st.text_input(
                    f"{selected_disp} API Key",
                    value=existing_key,
                    type="password",
                    placeholder=placeholder,
                    help=f"Can also be set via {env_key_name} environment variable.",
                    key=f"ui_api_key_{provider_key}",
                )

        base_url = ""
        if provider_key == "ollama":
            with col_url:
                base_url = st.text_input(
                    "Ollama / Endpoint URL",
                    value=llm_cfg.base_url or "http://localhost:11434/v1/chat/completions",
                    key="ui_ollama_base_url",
                )

        # Action Buttons: Fetch Models, Test Connection, Save
        c_fetch, c_test, c_save = st.columns([1, 1, 1])
        with c_fetch:
            if provider_key in ["groq", "openai", "ollama"]:
                if st.button("🔄 Fetch Live Models", use_container_width=True, key=f"fetch_btn_{provider_key}"):
                    if not api_key and provider_key != "ollama":
                        st.warning("Please enter your API Key first.")
                    else:
                        with st.spinner("Fetching active models from API..."):
                            profiler = LLMPersonProfiler(cfg)
                            live_models = profiler.fetch_available_models(provider_key, api_key, base_url)
                            if live_models:
                                st.session_state[cached_models_key] = live_models
                                st.success(f"Loaded {len(live_models)} active models!")
                                st.rerun()
                            else:
                                st.error("Could not fetch models. Verify API Key.")

        with c_test:
            if st.button("🧪 Test Connection", use_container_width=True, key=f"test_btn_{provider_key}"):
                with st.spinner(f"Testing {selected_disp} ({model_name})..."):
                    profiler = LLMPersonProfiler(cfg)
                    success, msg = profiler.test_connection(
                        provider=provider_key,
                        api_key=api_key,
                        model_name=model_name,
                        base_url=base_url,
                    )
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

        with c_save:
            if st.button("💾 Save Settings", use_container_width=True, type="primary", key=f"save_btn_{provider_key}"):
                llm_cfg.provider = provider_key
                llm_cfg.model_name = model_name
                llm_cfg.api_key = api_key
                if base_url:
                    llm_cfg.base_url = base_url

                # Save to DB if available
                if db is not None:
                    db.set_setting("llm.provider", provider_key)
                    db.set_setting("llm.model_name", model_name)
                    if api_key:
                        db.set_setting("llm.api_key", api_key)
                    if base_url:
                        db.set_setting("llm.base_url", base_url)

                # Set environment variable as well
                env_key_name = prov_info.get("key_env_var", "")
                if env_key_name and api_key:
                    os.environ[env_key_name] = api_key

                st.success(f"Settings saved! Using {selected_disp} ({model_name}) for video processing.")
