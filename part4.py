import streamlit as st
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
COHERE_API_KEY = "LvCz1r5ZjH0kOl9i7Zkn2MdBFmIzjJMT4X3ZAt5x"

st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt=st.text_input("enter your prompt here")
title_template=PromptTemplate(input_variables=["topic"], template="write me a youtube video title for {topic}")
script_template=PromptTemplate(input_variables=["title"], template="write me a script for the topic {title}")

memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

llm = Cohere(cohere_api_key=COHERE_API_KEY)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)

sequential_chain=SequentialChain(chains=[title_chain, script_chain],input_variables=["topic"], output_variables=["title","script"],verbose=True)


if prompt:
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])


    with st.expander('Message history'):
        st.info(memory.buffer)