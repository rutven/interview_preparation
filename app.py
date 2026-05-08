import os

import httpx
import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

load_dotenv()

DEFAULT_MODELS = {
    "mistral": "mistral-large-latest",
    "gemini": "gemini-2.5-flash",
    "openrouter": "deepseek/deepseek-v4-pro",
}


class OpenRouterChatModel:
    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key

    def invoke(self, messages):
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": message.content}
                    for message in messages
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


@st.cache_resource
def get_llm(provider, model):
    provider = provider.lower()

    if provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required when LLM_PROVIDER=mistral")
        return ChatMistralAI(model=model, api_key=api_key)

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(model=model, api_key=api_key)

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter"
            )
        return OpenRouterChatModel(model=model, api_key=api_key)

    supported_providers = ", ".join(DEFAULT_MODELS)
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use one of: {supported_providers}"
    )


def invoke_llm(messages):
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider))
    if not model:
        raise ValueError(f"LLM_MODEL is required for provider '{provider}'")

    llm = get_llm(provider, model)
    response = llm.invoke(messages)
    if isinstance(response, str):
        return response
    return response.content


st.title("Interview Coach")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
def get_feedback(question, answer):
    messages = [
        SystemMessage(f"""
                    You are an interview evaluator.
            Question: {question}
            Candidate's answer: {answer}

        Give brief feedback in 3-4 sentences max covering:
        - What was good
        - What was missing or incorrect
        - One specific improvement""")
    ]

    return invoke_llm(messages)


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
def get_next_question(role, difficulty, asked_questions=None):
    question_list = (
        f"\n Do not repeat these questions: {asked_questions}"
        if asked_questions
        else ""
    )
    messages = [
        SystemMessage(f"""You are an interview coach for {role} at {difficulty} level.
    Ask ONE interview question only.
    No explanations, no follow-up probes, no commentary.{question_list}
    Just the question.""")
    ]
    return invoke_llm(messages)


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
def get_summary(messages):
    history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    message = [SystemMessage(f"Create summary based on whole session {history}")]
    return invoke_llm(message)


if st.session_state.interview_started:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if user_input := st.chat_input(
        "Type your answer...", disabled=st.session_state.interview_finished
    ):
        question = st.session_state.messages[-1]["content"]
        answer = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating feedback in progress..."):
            try:
                feedback = get_feedback(question, answer)
            except Exception as e:
                print(f"Error type: {type(e).__name__}")
                feedback = f"⚠️ Feedback currently unavailable due to high traffic, but let's continue!{e}"
                st.error(
                    "The LLM provider is not responding. Please check your connection."
                )
        st.session_state.messages.append(
            {"role": "assistant", "content": feedback, "type": "feedback"}
        )
        asked_questions = [
            message["content"]
            for message in st.session_state.messages
            if message.get("type") == "question"
        ]
        if (
            st.session_state.question_count >= 5
            and not st.session_state.interview_finished
        ):
            with st.spinner("Generating summary in progress..."):
                try:
                    whole_summary = get_summary(st.session_state.messages)
                except Exception as e:
                    print(f"Error type: {type(e).__name__}")
                    whole_summary = f"🏁 Interview finished! I couldn't generate a summary due to high traffic, but thanks for participating. {e}"
                    st.warning("Could not generate summary, finishing session.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": whole_summary}
                )
                st.session_state.interview_finished = True
                st.rerun()
        else:
            with st.spinner("Generating question in progress..."):
                try:
                    next_question = get_next_question(
                        st.session_state.role,
                        st.session_state.difficulty,
                        asked_questions,
                    )
                except Exception as e:
                    print(f"Error type: {type(e).__name__}")
                    next_question = "I'm having trouble connecting. Could you please try to refresh or wait a moment?"
                    st.error(
                        "The LLM provider is not responding. Please check your connection."
                    )
            st.session_state.messages.append(
                {"role": "assistant", "content": next_question, "type": "question"}
            )
            st.session_state.question_count += 1
            st.rerun()
else:
    role = st.selectbox(
        "Pick a role: ",
        [
            "Frontend Developer",
            "Backend Developer",
            "Data Analyst",
            "QA Engineer",
            "DevOps Engineer",
        ],
    )
    difficulty = st.radio("Pick a difficulty: ", ["Junior", "Mid", "Senior"])
    start_button = st.button("Start an interview")
    if start_button:
        st.session_state.interview_started = True
        st.session_state.role = role
        st.session_state.difficulty = difficulty
        with st.spinner("Generating question in progress..."):
            try:
                question = get_next_question(
                    st.session_state.role, st.session_state.difficulty
                )
            except Exception as e:
                print(f"Error type: {type(e).__name__}")
                question = "I'm having trouble connecting. Could you please try to refresh or wait a moment?"
                st.error(
                    "The LLM provider is not responding. Please check your connection."
                )
        st.session_state.messages.append(
            {"role": "assistant", "content": question, "type": "question"}
        )
        st.session_state.question_count += 1
        st.rerun()


def reset_interview():
    st.session_state.interview_started = False
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.interview_finished = False
    print(st.session_state)
    if "role" in st.session_state:
        del st.session_state.role
    if "difficulty" in st.session_state:
        del st.session_state.difficulty


if st.session_state.get("interview_finished", False):
    st.write("---")
    st.success("Interview finished!!")
    st.button("Start New Interview", on_click=reset_interview)
