import json
import logging
from pathlib import Path

import streamlit as st  # type: ignore
from google import genai  # type: ignore  (new unified Google GenAI SDK)
from google.genai import types  # type: ignore

#-----------------------------------
# LOGGING (console log: selected model, model sent, response status, errors)
#-----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("gemini_chatbot")

#-----------------------------------
#PAGE CONFIG
#-----------------------------------
st.set_page_config(
    page_title="Google Gemini AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Google Gemini AI Chatbot")

# --- Requirement 1/2/12: EXACTLY these four models. No dynamic discovery. ---
SUPPORTED_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]
FALLBACK_MODEL = "gemini-flash-latest"  # only ever used if fallback is explicitly enabled AND selected model fails

# --- Requirement 9: persist model selection across app restarts. ---
# NOTE: Standard Streamlit has no access to the browser's real localStorage
# without a custom JS component. "Restored on application startup" is
# implemented here via a small local config file on disk instead. This
# stores ONLY the model name -- never the API key.
CONFIG_PATH = Path.home() / ".gemini_chatbot" / "config.json"


def load_saved_model() -> str:
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            saved = data.get("selected_model")
            if saved in SUPPORTED_MODELS:
                return saved
    except Exception as e:
        logger.warning(f"Could not read saved model preference: {e}")
    return SUPPORTED_MODELS[0]


def save_selected_model(model_name: str):
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps({"selected_model": model_name}))
    except Exception as e:
        logger.warning(f"Could not persist model preference: {e}")


#-----------------------------------
# SESSION STATE
#-----------------------------------
defaults = {
    "api_valid": False,
    "api_key": "",
    "client": None,
    "chat_history": [],
    "chat_session": None,
    "selected_model": load_saved_model(),   # restored on startup
    "system_prompt": "",
    "fallback_enabled": True,               # Requirement 7/8: must be explicit, but user-controlled
    "request_log": [],
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


#-----------------------------------
# HELPERS
#-----------------------------------
def log_event(model_selected: str, model_sent: str, status: str, detail: str = ""):
    """Requirement 11: detailed console logging of selected model, model sent,
    response status, and any model-related errors."""
    line = (
        f"selected_model=`{model_selected}` | model_sent_to_api=`{model_sent}` "
        f"| status={status}" + (f" | detail={detail}" if detail else "")
    )
    logger.info(line)
    st.session_state.request_log.append(line)
    st.session_state.request_log = st.session_state.request_log[-30:]


def genai_history_from_chat_history():
    history = []
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [{"text": msg["content"]}]})
    return history


def build_chat_config(persona: str):
    if persona.strip():
        return types.GenerateContentConfig(system_instruction=persona.strip())
    return None


def rebuild_chat_session(model_name: str, persona: str, keep_history: bool = True):
    """Requirement 3/4: the exact model passed in is the exact model used -- no override."""
    history = genai_history_from_chat_history() if keep_history else []
    st.session_state.chat_session = st.session_state.client.chats.create(
        model=model_name,
        config=build_chat_config(persona),
        history=history,
    )


def call_gemini(action: str, model_name: str, persona: str, prompt: str = None):
    """
    Single entry point for every Gemini call (validate + chat).
    - `model_name` is ALWAYS the exact value passed by the caller (the dropdown
      selection) -- there is no hardcoded override anywhere in this function.
    - Requirement 6: model_name is checked against SUPPORTED_MODELS before sending.
    - Requirement 7: on failure, reports clearly which model failed and does NOT
      silently switch unless st.session_state.fallback_enabled is True.
    - Requirement 8: fallback (if enabled) is ONLY attempted after a real failure,
      never proactively, and is always gemini-flash-latest.
    Returns: (success: bool, message_or_reply: str, model_actually_used: str, used_fallback: bool)
    """
    # Requirement 6: validate before sending
    if model_name not in SUPPORTED_MODELS:
        log_event(model_name, "NONE", "REJECTED", "model not in SUPPORTED_MODELS")
        return False, f"❌ `{model_name}` is not one of the supported models: {SUPPORTED_MODELS}", model_name, False

    def _attempt(m_name):
        log_event(model_name, m_name, "REQUEST_SENT", f"action={action}")
        if action == "validate":
            resp = st.session_state.client.models.generate_content(
                model=m_name,
                contents="Say 'API Connected Successfully'",
            )
            return resp.text
        elif action == "chat":
            rebuild_chat_session(m_name, persona, keep_history=True)
            resp = st.session_state.chat_session.send_message(prompt)
            return resp.text
        else:
            raise ValueError(f"Unknown action: {action}")

    try:
        result = _attempt(model_name)
        log_event(model_name, model_name, "SUCCESS")
        return True, result, model_name, False
    except Exception as e:
        err_str = str(e)
        log_event(model_name, model_name, "FAILED", err_str)

        if not st.session_state.fallback_enabled:
            # Requirement 7: do NOT switch models unless fallback is explicitly enabled.
            return False, f"❌ Model `{model_name}` failed and automatic fallback is turned off. Error: {err_str}", model_name, False

        if model_name == FALLBACK_MODEL:
            return False, f"❌ `{model_name}` failed and it is already the fallback model. Error: {err_str}", model_name, False

        st.warning(f"⚠️ Model `{model_name}` failed: {err_str[:150]}. Fallback is enabled — retrying with `{FALLBACK_MODEL}`...")
        try:
            result = _attempt(FALLBACK_MODEL)
            log_event(model_name, FALLBACK_MODEL, "FALLBACK_SUCCESS")
            return True, result, FALLBACK_MODEL, True
        except Exception as e2:
            err2 = str(e2)
            log_event(model_name, FALLBACK_MODEL, "FALLBACK_FAILED", err2)
            return False, f"❌ Both `{model_name}` and fallback `{FALLBACK_MODEL}` failed. Error: {err2}", FALLBACK_MODEL, True


#-----------------------------------
# SIDEBAR: SETTINGS
#-----------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Model")
    new_model = st.selectbox(
        "Choose a Gemini model",
        SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(st.session_state.selected_model),
        help="These are the only four supported models.",
        key="model_dropdown",
    )

    st.subheader("🔁 Fallback")
    fallback_enabled = st.checkbox(
        f"If the selected model fails, retry with `{FALLBACK_MODEL}`",
        value=st.session_state.fallback_enabled,
        help="Fallback only ever fires AFTER a real failure. It is never used proactively."
    )

    st.subheader("Persona / System Prompt")
    new_persona = st.text_area(
        "Give the assistant a personality or instructions",
        value=st.session_state.system_prompt,
        placeholder="e.g. You are a friendly Python tutor who explains things simply.",
        height=120
    )

    settings_changed = (
        new_model != st.session_state.selected_model
        or new_persona != st.session_state.system_prompt
        or fallback_enabled != st.session_state.fallback_enabled
    )

    if st.button("✅ Apply Settings", disabled=not st.session_state.api_valid):
        st.session_state.selected_model = new_model
        st.session_state.system_prompt = new_persona
        st.session_state.fallback_enabled = fallback_enabled
        save_selected_model(new_model)  # Requirement 9: persist to disk
        rebuild_chat_session(new_model, new_persona, keep_history=True)
        st.success(f"Settings applied. Active model: `{new_model}`.")
        st.rerun()

    if settings_changed and st.session_state.api_valid:
        st.caption("⚠️ You have unapplied changes above.")

    st.divider()

    st.subheader("💾 Chat History")

    chat_json = json.dumps(st.session_state.chat_history, indent=2)
    st.download_button(
        "Download Chat History (.json)",
        data=chat_json,
        file_name="gemini_chat_history.json",
        mime="application/json",
        disabled=len(st.session_state.chat_history) == 0
    )

    uploaded_file = st.file_uploader("Load Chat History", type=["json"])
    if uploaded_file is not None:
        if st.button("📂 Restore this chat"):
            try:
                loaded_history = json.load(uploaded_file)
                if isinstance(loaded_history, list):
                    st.session_state.chat_history = loaded_history
                    if st.session_state.api_valid:
                        rebuild_chat_session(st.session_state.selected_model, st.session_state.system_prompt, keep_history=True)
                    st.success("Chat history restored.")
                    st.rerun()
                else:
                    st.error("That file doesn't look like a valid chat history export.")
            except Exception as e:
                st.error(f"Couldn't load file: {str(e)}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.api_valid:
            rebuild_chat_session(st.session_state.selected_model, st.session_state.system_prompt, keep_history=False)
        st.rerun()

    st.divider()
    with st.expander("🔍 Debug Log (model calls)"):
        if st.session_state.request_log:
            st.code("\n".join(st.session_state.request_log), language="text")
        else:
            st.caption("No requests made yet.")

#-----------------------------------
# CONNECTION STATUS  (Requirement 5: active model always visible)
#-----------------------------------
st.subheader(" 🔌Connection Status")
if st.session_state.api_valid:
    st.success(f" 🟢Connected to Google Gemini AI API — **Active model: `{st.session_state.selected_model}`**")
else:
    st.error(" 🔴Not connected to Google Gemini AI API")

st.divider()

#-----------------------------------
#API KEY SECTION
#-----------------------------------
st.subheader(" 🔑 Step 1: Enter Your Google AI Studio API Key")
api_key = st.text_input(
    "Paste your Gemini API Key",
    type="password",
    placeholder="rTyu7Inz...."
)

if st.button("Validate API Key"):
    if not api_key:
        st.error("Please enter a valid API key.")
    else:
        st.session_state.api_key = api_key
        st.session_state.client = genai.Client(api_key=api_key)

        # Requirement 1/3: validate against the dropdown's CURRENT value, not a stale one.
        model_to_validate = new_model

        success, message, model_used, used_fallback = call_gemini(
            action="validate",
            model_name=model_to_validate,
            persona=st.session_state.system_prompt,
        )

        if success:
            st.session_state.api_valid = True
            st.session_state.selected_model = model_used
            st.session_state.fallback_enabled = fallback_enabled
            save_selected_model(model_used)
            st.session_state.chat_history = []
            rebuild_chat_session(model_used, st.session_state.system_prompt, keep_history=False)
            if used_fallback:
                st.info(f"ℹ️ `{model_to_validate}` failed, so fallback `{model_used}` was used instead (fallback was enabled).")
            st.success(f"✅ API Key is valid and connected using `{model_used}`.")
            st.rerun()
        else:
            st.session_state.api_valid = False
            st.session_state.client = None
            st.error(message)

st.divider()

#-----------------------------------
#CHATBOT SECTION
#-----------------------------------
if st.session_state.api_valid:
    st.subheader(" 💬 Step 2: Chat with Gemini")
    st.caption(f"Active model: `{st.session_state.selected_model}`")

    if st.session_state.chat_session is None:
        rebuild_chat_session(st.session_state.selected_model, st.session_state.system_prompt, keep_history=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your message here...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                success, reply_text, model_used, used_fallback = call_gemini(
                    action="chat",
                    model_name=st.session_state.selected_model,
                    persona=st.session_state.system_prompt,
                    prompt=user_prompt,
                )
                if success and used_fallback:
                    st.info(f"ℹ️ `{st.session_state.selected_model}` failed for this message. Fallback `{model_used}` was used (fallback is enabled) — your model setting stays on `{st.session_state.selected_model}` for next time.")
            st.markdown(reply_text)

        st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
else:
    st.info("👆 Please validate your API key above to start chatting.")