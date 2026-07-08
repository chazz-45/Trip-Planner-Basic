"""
Trip Planner Agent - Streamlit Frontend
------------------------------------------
Install dependencies:
    pip install camel-ai streamlit --break-system-packages

Set your API key before running:
    export OPENAI_API_KEY="your-key-here"        (Mac/Linux)
    setx OPENAI_API_KEY "your-key-here"           (Windows)

Run:
    streamlit run trip_planner_app.py

This will open automatically in your browser at http://localhost:8501
"""

import streamlit as st
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType


# ---------- Agent setup ----------

@st.cache_resource
def build_trip_planner_agent():
    """Creates the CAMEL ChatAgent once and reuses it across reruns."""
    model = ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=ModelType.GEMINI_2_5_FLASH,
        model_config_dict={"temperature": 0.7},
    )

    system_message = BaseMessage.make_assistant_message(
        role_name="Trip Planner",
        content=(
            "You are an expert travel planning assistant. "
            "Given a destination, number of days, budget, and interests, "
            "you create a clear, day-by-day itinerary including suggested "
            "activities, approximate costs, food recommendations, and travel "
            "tips. Keep responses well-structured with headings for each day."
        ),
    )

    return ChatAgent(system_message=system_message, model=model)


def ask_agent(agent: ChatAgent, prompt: str) -> str:
    user_message = BaseMessage.make_user_message(role_name="Traveler", content=prompt)
    response = agent.step(user_message)
    return response.msgs[0].content


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Chazz Trip Planner", page_icon="🧳", layout="centered")

st.title("🧳 Chazz Trip Planner")
st.caption("Powered by Tech Titans")

# Session state to hold chat history and the agent
if "agent" not in st.session_state:
    st.session_state.agent = build_trip_planner_agent()
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text) tuples
if "itinerary_generated" not in st.session_state:
    st.session_state.itinerary_generated = False

# --- Trip details form ---
with st.form("trip_form"):
    col1, col2 = st.columns(2)
    with col1:
        destination = st.text_input("Destination", placeholder="e.g. Goa, India")
        days = st.number_input("Number of days", min_value=1, max_value=60, value=3)
    with col2:
        budget = st.selectbox("Budget", ["Budget", "Moderate", "Luxury"])
        interests = st.text_input("Interests", placeholder="e.g. food, history, beaches")

    submitted = st.form_submit_button("Generate Itinerary")

if submitted:
    if not destination.strip():
        st.warning("Please enter a destination.")
    else:
        prompt = (
            f"Plan a {days}-day trip to {destination}.\n"
            f"Budget: {budget}\n"
            f"Interests: {interests or 'general sightseeing'}\n"
            f"Please give a day-by-day itinerary with activities, rough costs, "
            f"and any useful local tips."
        )
        with st.spinner("Planning your trip..."):
            itinerary = ask_agent(st.session_state.agent, prompt)

        st.session_state.history.append(("user", f"Plan a trip to {destination} ({days} days, {budget}, interests: {interests})"))
        st.session_state.history.append(("assistant", itinerary))
        st.session_state.itinerary_generated = True

# --- Display conversation ---
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

# --- Follow-up chat, only shown after first itinerary ---
if st.session_state.itinerary_generated:
    follow_up = st.chat_input("Ask a follow-up, e.g. 'make day 2 cheaper'")
    if follow_up:
        st.session_state.history.append(("user", follow_up))
        with st.spinner("Updating..."):
            reply = ask_agent(st.session_state.agent, follow_up)
        st.session_state.history.append(("assistant", reply))
        st.rerun()
