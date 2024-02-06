import os
import tempfile
import streamlit as st
from streamlit_chat import message
import yaml
from pathlib import Path
from .chat import ChatPDF
from .utils import Embeddings

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(
            f"Ingesting {file.name}"
        ):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def update_model():
    st.session_state["assistant"] = ChatPDF(
        model=st.session_state["model"],
        embeddings=Embeddings[st.session_state["embeddings"]].value,
    )


def page():
    with open(Path("config/config.yaml"), "r") as configfile:
        config = yaml.safe_load(configfile)
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["model"] = config["models"][0]
        st.session_state["embeddings"] = config["embeddings"][0]
        st.session_state["assistant"] = ChatPDF(
            model=st.session_state["model"],
            embeddings=Embeddings[st.session_state["embeddings"]].value,
        )

    st.header("ChatPDF")

    # It only displays the select boxes if the options are greater than 1
    if len(config["models"]) > 1 or len(config["embeddings"]) > 1:
        st.subheader("Select a model")
        if len(config["models"]) > 1:
            st.session_state["model"] = st.selectbox(
                label="Select a model", options=config["models"]
            )
        if len(config["embeddings"]) > 1:
            st.session_state["embeddings"] = st.selectbox(
                label="Select an embeddings", options=config["embeddings"]
            )
        st.button(label="Apply", on_click=update_model)

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)
    if st.checkbox("debug"):
        st.write(st.session_state)
