from typing import Dict, List
import streamlit as st
import requests

st.title("Parla con Velvet")

BASE_URL: str = "http://localhost:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_history() -> List[Dict[str, str]]:
    return [
        {
            "role": "user" if message["role"] == "user" else "assistant",
            "message": message["content"],
        }
        for message in st.session_state.messages
    ]


def main() -> None:

    # Mostra i messaggi precedenti
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Scrivi il tuo messaggio e premi invio...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history: List[Dict[str, str]] = get_history()

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response_text = ""

            with requests.post(
                url=f"{BASE_URL}/api/v1/chat",
                json={"message": prompt, "history": history},
                stream=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        text_chunk = chunk.decode("utf-8")
                        response_text += text_chunk
                        message_placeholder.markdown(
                            response_text + "▌"
                        )  # Simula digitazione

            message_placeholder.markdown(response_text)  # Mostra il testo completo
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )


if __name__ == "__main__":
    main()
