import streamlit as st
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
COHERE_API_KEY = "LvCz1r5ZjH0kOl9i7Zkn2MdBFmIzjJMT4X3ZAt5x"

st.title("Youtube Video Script Generator with LangChain 🦜🔗")

prompt=st.text_input("enter your prompt here")
title_template=PromptTemplate(input_variables=["topic"],
                              template="write me a youtube video title for {topic}")
script_template=PromptTemplate(input_variables=["title","wikipedia_research"],
                               template="write me a script for the topic {title} while taking the help of "
                                        "wikipedia{wikipedia_research}")

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

llm = Cohere(cohere_api_key=COHERE_API_KEY)
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki= WikipediaAPIWrapper()


if prompt:
    title=title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    script=script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('title history'):
        st.info(title_memory.buffer)
    with st.expander('script history'):
        st.info(script_memory.buffer)
    with st.expander('wikipedia history'):
        st.info(wiki_research)