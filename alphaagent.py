
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.utilities import WolframAlphaAPIWrapper, GoogleSearchAPIWrapper
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
import pandas as pd
import streamlit as st
import json
import openai


########### Setup api Keys ###################################
openai.api_key = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
WOLFRAM_ALPHA_APPID = st.secrets["WOLFRAM_ALPHA_APPID"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
###############################################################

###############################################################
# Set this to your Zep server URL
#ZEP_API_URL = "http://localhost:8000"
#ZEP_API_KEY = "<your JWT token>" # optional

#session_id = str(uuid4())  # This is a unique identifier for the user

# Set up ZepMemory instance
#memory = ZepMemory(
#    session_id=session_id,
#    url=ZEP_API_URL,
#    memory_key="chat_history",
#)
###################################################################


####################### Create Memory ####################################
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
##########################################################################



############ Create LLMs (Language Models) / Agents / Special Tools  #######################
wolfram = WolframAlphaAPIWrapper()
search = GoogleSearchAPIWrapper()
llm = ChatOpenAI(temperature=0.5, model="gpt-4o", streaming = True)
#llm3 = ChatOpenAI(temperature = 0.6, model= "gpt-4-0613", streaming= True)
llm2 = ChatOpenAI(temperature = 0, model= "gpt-3.5-turbo-0613", streaming= True)
#email_agent = initialize_agent(toolkit.get_tools(), llm2, agent= AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True,  handle_parsing_errors = True, memory = memory)
#calendar_agent =  initialize_agent(toolkit.get_tools(), llm2, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose = True, handle_parsing_errors = True, memory = memory)
###########################################################################################

def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    llm = llm2

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=True)


def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.markdown(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.dataframe(df)


####################### Function to move focus to chat input ################################
def move_focus():
    # inspect the html to determine which control to specify to receive focus (e.g. text or textarea).
    st.components.v1.html(
        f"""
            <script>
                var textarea = window.parent.document.querySelectorAll("textarea[type=textarea]");
                for (var i = 0; i < textarea.length; ++i) {{
                    textarea[i].focus();
                }}
            </script>
        """,
    )
#################################################################################################

    
################ Function to complete messages using LLM  ##################
def complete_messages(nbegin, nend, stream=False):
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    with st.spinner(f"Waiting for {nbegin}/{nend} responses from Alpha."):
        if stream:
            responses = []
            for message in messages:
                prompt = message["content"]
                assistant_content = generate_response(prompt)  # Use your LLM to generate response
                responses.append(assistant_content)
            response_content = "".join(responses)
        else:
            response = generate_response(messages[-1]["content"])  # Use LLM to generate response
            response_content = response
    return response_content
##############################################################################################
#####################################################################



######################### Agent Toolkit #################################

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

###### Updated Function Calling and Tool Bindings for Langchain / OpenAI ###
#llm_with_tools = llm.bind(functions = [format_tool_to_openai_function(t) for t in tools])
#llm_with_tools2 = llm2.bind(functions = [format_tool_to_openai_function(t) for t in tools])
############# Generate Agent Response With Conversational Langchain Agent ################
def generate_response(prompt):
    agent1 = initialize_agent(
            tools, 
            llm, 
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose = True, #handle_parsing_errors = True, 
            memory = memory)
    message = agent1(prompt)
    return message["output"]
#
    
#########################################################################################

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data or an empty dictionary if decoding fails"""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Handle JSON decoding error here, you can log the error or return an empty dictionary
        print(f"JSON decoding error: {e}")
        return {}
####################################################################

    


