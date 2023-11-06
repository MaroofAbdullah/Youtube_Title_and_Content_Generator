import streamlit as st
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
COHERE_API_KEY = "LvCz1r5ZjH0kOl9i7Zkn2MdBFmIzjJMT4X3ZAt5x"

st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt=st.text_input("Plug in your prompt here")

title_template = PromptTemplate(input_variables=['topic'], template="write me a youtube video title for {topic}")
script_template = PromptTemplate(input_variables=['title'], template="write me a script for the topic {title}")

llm = Cohere(cohere_api_key=COHERE_API_KEY)

title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True)

sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)


if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)

