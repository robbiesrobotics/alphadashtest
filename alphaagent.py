from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.utilities import WolframAlphaAPIWrapper, GoogleSearchAPIWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
import pandas as pd
import streamlit as st
import json
import openai

########### Setup api Keys ###################################
ZAPIER_CLIENT_SECRET = st.secrets["ZAPIER_CLIENT_SECRET"]
ZAPIER_CLIENT_ID = st.secrets["ZAPIER_CLIENT_ID"]
ZAPIER_NLA_API_KEY = st.secrets["ZAPIER_NLA_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
WOLFRAM_ALPHA_APPID = st.secrets["WOLFRAM_ALPHA_APPID"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
###############################################################

################## Create Agent's Conversational Buffer Memory ###########################
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
##########################################################################################


############ Create LLMs (Language Models) / Agents / Special Tools  #######################
zapier = ZapierNLAWrapper()
wolfram = WolframAlphaAPIWrapper()
search = GoogleSearchAPIWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
llm = ChatOpenAI(temperature=0.7,model="gpt-4o", streaming = True)
llm2 = OpenAI(temperature = 0, streaming= True)
email_agent = initialize_agent(toolkit.get_tools(), llm2, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True,  handle_parsing_errors = True, memory = memory)
calendar_agent =  initialize_agent(toolkit.get_tools(), llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose = True, handle_parsing_errors = True, memory = memory)
###########################################################################################



###################### TOOLS #####################################

tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for answering questions about current events, people, places."
    ),
    Tool(
        name="Wolfram Search",
        func=wolfram.run,
        description="Useful for answering questions about math, science, weather, date, and time."
    ),
]
###########################################################################################


############# Generate Agent Response With Conversational Langchain Agent ################
def generate_response(prompt):
    agent1 = initialize_agent(
            tools, 
            llm, 
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose = True, handle_parsing_errors = True, 
            memory = memory)
    message = agent1(prompt)
    return message["output"]
#########################################################################################


