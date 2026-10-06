from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()

template = load_prompt("template.json")

llm = HuggingFaceEndpoint(
     repo_id="openai/gpt-oss-120b",
    task= "text-generation"
)


model = ChatHuggingFace(llm=llm)

st.header('Research Tool')

paper_input = st.selectbox(
    "Select a research paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Language Models are Few-Shot Learners",
        "Deep Residual Learning for Image Recognition",
        "Generative Adversarial Networks",
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "You Only Look Once: Unified, Real-Time Object Detection",
        "Adam: A Method for Stochastic Optimization"
    ]
)

style_input = st.selectbox(
    "Select explanation style",
    [
        "Beginner-Friendly",
        "Technical",
        "Academic",
        "Simplified"
    ]
)

length_input = st.selectbox(
    "Select summary length",
    [
        "Short (100 words)",
        "Medium (250 words)",
        "Long (500 words)"
    ]
)



if st.button('summarise'):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write(result.content)