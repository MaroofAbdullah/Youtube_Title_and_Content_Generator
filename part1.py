# import streamlit as st
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate
# import os
# os.environ["OPENAI_API_KEY"]='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'
#
# st.title("Youtube Video Script Generator with LangChain 🦜🔗")
#
#
# prompt=st.text_input("Plug in your prompt here")
# llm=OpenAI(temperature=0.9)
#
# if prompt:
#     response = llm(prompt)
#     st.write(response)

import streamlit as st
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
COHERE_API_KEY = "LvCz1r5ZjH0kOl9i7Zkn2MdBFmIzjJMT4X3ZAt5x"

st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt1=st.text_input("Plug in your prompt here")

template = """Question: give me youtube video title for {prompt1}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["prompt1"])

llm = Cohere(cohere_api_key=COHERE_API_KEY)

llm_chain = LLMChain(prompt=prompt, llm=llm)


if prompt:
    response = llm_chain.run(prompt1)
    st.write(response)