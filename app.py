import os
import streamlit as st
from agents.therapist import get_therapist_chain
from agents.mindfulness import get_mindfulness_chain
from agents.knowledge import get_knowledge_chain
from agents.journal import get_journal_chain
from router import route_user_input

# Set your Gemini API key from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # For local development, you can still use this, but comment it out for deployment
    # os.environ["GOOGLE_API_KEY"] = "YOUR_KEY_HERE"
    st.error("API key not found in secrets. Please configure it in your Streamlit Cloud dashboard.")

# Streamlit page config
st.set_page_config(page_title="Tranquil AI", layout="centered")
st.title("🍃 Tranquil")
st.write("Hey, I'm Tranquil. Let's untangle your thoughts.")

# Initialize agents
if "agents" not in st.session_state:
    st.session_state.agents = {
        "therapist": get_therapist_chain(),
        "mindfulness": get_mindfulness_chain(),
        "knowledge": get_knowledge_chain(),
        "journal": get_journal_chain()
    }

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["ai"])

# Input box
user_input = st.chat_input("How can I help you today?")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.status("Selecting the right agent for you...", expanded=False) as status:
        # Let the AI decide which agent to use
        agent_key = route_user_input(user_input)
        status.update(label=f"Using {agent_key.title()} agent to respond...", state="running")
        
        # Get response from the chosen agent
        agent = st.session_state.agents[agent_key]
        response = agent.run(user_input)
        
        status.update(label=f"Response from {agent_key.title()} agent ready!", state="complete")
    
    # Format the response with agent type
    formatted_response = f"**[{agent_key.title()} Agent]:** {response}"
    
    with st.chat_message("assistant"):
        st.markdown(formatted_response)

    # Save message
    st.session_state.messages.append({"user": user_input, "ai": formatted_response})