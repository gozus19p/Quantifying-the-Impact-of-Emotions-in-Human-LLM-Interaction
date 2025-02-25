from typing import Dict, List
import streamlit as st
import requests

st.title(f"Parla con Velvet")

BASE_URL: str = "http://localhost:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_history() -> List[Dict[str, str]]:

    return [
        {
            "role": "Utente" if message["role"] == "user" else "Assistente",
            "message": message["content"],
        }
        for message in st.session_state.messages
    ]


def main() -> None:
    prompt = st.chat_input("Scrivi il tuo messaggio e premi invio...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        history: List[Dict[str, str]] = get_history()

        response: requests.Response = requests.post(
            url=f"{BASE_URL}/api/v1/chat", json={"message": prompt, "history": history}
        )
        response.raise_for_status()
        answer: str = response.json()["message"]
        st.session_state.messages.append({"role": "ai", "content": answer})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == "__main__":
    main()
