import os

import streamlit as st

from dashboard.event_client import EventClient

st.set_page_config(page_title="Offline Voice Stack", layout="wide")

EVENT_BUS_URI = os.environ.get("EVENT_BUS_URI", "ws://localhost:8765")

if "event_client" not in st.session_state:
    st.session_state.event_client = EventClient(EVENT_BUS_URI)
    st.session_state.event_client.start()

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = "unknown"

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "assistant_reply" not in st.session_state:
    st.session_state.assistant_reply = ""

if "latencies" not in st.session_state:
    st.session_state.latencies = {}


def apply_event(event: dict) -> None:
    event_type = event.get("type")

    if event_type == "state":
        st.session_state.pipeline_state = event.get("value", "unknown")
    elif event_type == "wake_word":
        st.session_state.pipeline_state = f"wake word: {event.get('name')}"
    elif event_type == "transcript":
        st.session_state.transcript = event.get("text", "")
        st.session_state.assistant_reply = ""
    elif event_type == "reply_partial":
        st.session_state.assistant_reply = event.get("text", "")
    elif event_type == "assistant_reply":
        st.session_state.assistant_reply = event.get("text", "")
    elif event_type == "interrupted":
        st.session_state.pipeline_state = "interrupted"
    elif event_type == "latency":
        st.session_state.latencies[event.get("stage")] = event.get("seconds")


st.title("Offline Voice Stack")

STATE_LABELS = {
    "waiting_for_wake_word": "Idle, waiting for wake word",
    "recording": "Listening",
    "speaking": "Speaking",
    "interrupted": "Interrupted",
}


@st.fragment(run_every="0.5s")
def live_view() -> None:
    for event in st.session_state.event_client.drain():
        apply_event(event)

    label = STATE_LABELS.get(st.session_state.pipeline_state, st.session_state.pipeline_state)
    st.subheader(f"Status: {label}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**You said**")
        st.write(st.session_state.transcript or "-")

    with col2:
        st.markdown("**Assistant**")
        st.write(st.session_state.assistant_reply or "-")

    st.markdown("**Latency (seconds)**")
    if st.session_state.latencies:
        st.table(st.session_state.latencies)
    else:
        st.write("no data yet")


live_view()