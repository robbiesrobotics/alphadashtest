############ import dependencies for application to run ########################
import openai
import base64
import requests
import json
from io import BytesIO
import wolframalpha
import streamlit as st
from bs4 import BeautifulSoup
import os
import PyPDF2
import pandas as pd
from PIL import Image
import openpyxl
import database as db
import streamlit_authenticator as stauth
import re  # Import the regular expressions module
from alphaagent import initialize_agent # Import initialize_agent from the secondary app
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool  # Import AgentType from the secondary app
from langchain.memory import ConversationBufferMemory # Import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.utilities import GoogleSearchAPIWrapper, WolframAlphaAPIWrapper  # Import search tools
##################################################################################

######## Streamlit Page Config ####################################
st.set_page_config(
    page_title="AlphaDash",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)
##################################################################


############# USER Authentication Creation ###########################
users = db.fetch_all_users()
usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "AlphaChat", "abcdef", cookie_expiry_days=1)
name, authentication_status, username = authenticator.login("Login", "sidebar")
######################################################################

############### Authentication Status Conditionals ################
if st.session_state["authentication_status"]:
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')  
#####################################################################


############### Initialize LLM For Google Search and other tools ############################
llm = ChatOpenAI(temperature=0.5, model="gpt-4o", streaming = True)

#############################################################################################

################### Initialize messages in session_state if not present ######################################
if "messages" not in st.session_state:
    st.session_state.messages = []
################################################################################################

######################## Initialize your API keys and clients ##################################
st.title("AlphaDash Chat Assistant")
openai.api_key = st.secrets["OPENAI_API_KEY"]
wolfram_client = wolframalpha.Client(st.secrets["WOLFRAM_ALPHA_APPID"])
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GOOGLE_CX = st.secrets["GOOGLE_CSE_ID"]
num_images = 1
image_size = "1024x1024"
quality = "hd"
################################################################################################

########################### Initialize Google Search tool ######################################
search = GoogleSearchAPIWrapper()
################################################################################################


########################## Define Google Search Tool for Langchain Use ######################
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for answering questions about current events, people, places."
    ),
]
##############################################################################################


####################### Create Agent and Assistant Short-term Memory ####################################
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
##############################################################################################

############################# Initialize Langchain Agent ########################################
agent1 = initialize_agent(
    tools,
    llm,  # assuming llm is already initialized in your primary app
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
###############################################################################################


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Function to handle text files
def handle_txt(file):
    text = file.read().decode("utf-8")
    return f"Text file processed. First 500 characters:\n{text[:500]}"

# Function to handle CSV files
def handle_csv(file):
    df = pd.read_csv(file)
    return f"CSV file processed. {df.shape[0]} rows and {df.shape[1]} columns."
        
        
# Function to handle PDF files
def handle_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    return f"PDF file processed. {num_pages} pages."

# Function to handle Excel files
def handle_excel(file, extension):
    df = pd.read_excel(file, engine='openpyxl' if extension == '.xlsx' else 'xlrd')
    return f"Excel file processed. {df.shape[0]} rows and {df.shape[1]} columns."

# Function to handle image files
def handle_image(file):
    image = Image.open(file)
    return f"Image file processed. Dimensions: {image.size[0]} x {image.size[1]} pixels."

#Function to read the content of a webpage        
def read_webpage_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

#Function or Langchain Google Search Tool
def googleSearchFunction(query):
    message = agent1(query)  # Use the initialized agent to generate a response
    return message["output"]  # Assuming the agent returns a dictionary with an 'output' key

# Function to get a refined answer from OpenAI based on Google's result
def get_refined_answer_from_openai(original_query, google_result, conversation_history):
    messages = conversation_history + [
        {"role": "user", "content": original_query},
        {"role": "assistant", "content": f"Here is some information I found: {google_result}"}
    ]
    
    # Initialize an empty string for the refined_answer
    refined_answer = ""

    # Call OpenAI API and update the message_placeholder as we get more content
    for response in openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=st.session_state.messages + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Here is some information I found: {google_result}"}
        ],
        stream=True
    ):
        refined_answer += response.choices[0].delta.get("content", "")
        full_response = refined_answer
        message_placeholder.markdown(full_response + "▌")

    
    return refined_answer

# Define URL pattern at a global scope
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

# Wolfram Alpha Function
def wolframFunction(query):
    res = wolfram_client.query(query)
    answer = next(res.results).text
    return answer

# Function to interact with the DALL·E API for image generation
def generate_images(prompt):
    response = openai.Image.create(
        model="dall-e-3",
        prompt=prompt,
        n=num_images,
        size=image_size,
        quality="hd",
    )

    image_data = []

    # Check if the response contains image data and extract them
    if 'data' in response and len(response['data']) > 0:
        for data_item in response['data']:
            if 'url' in data_item:
                # For Streamlit to display the image, download the image data into memory
                image_response = requests.get(data_item['url'])
                image_data.append(Image.open(BytesIO(image_response.content)))

    if image_data:
        for img in image_data:
            st.image(img, caption=prompt)  # Display the image using Streamlit
            st.session_state.messages.append({
                "role": "image",
                "content": img
            })
        return img
    else:
        st.error("Image generation failed")
        return []

        
# Function to determine which API to use based on the query
def get_api_choice(query):
    # Check for URLs in the query
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    read_webpage_keywords = ['link', 'read this', 'summarize this', 'website', 'webpage', url_pattern]
    wolfram_keywords = ['current date', 'wolfram', 'is today','todays date','math','what day', 'time', 'weather', 'day of the week', r'\b\d+\s*\+\s*\d+\b', r'\d+\s*-\s*\d+', r'\d+\s*\*\s*\d+', r'\d+\s*/\s*\d+']
    google_keywords = ['current event', 'news', 'who is', 'google', 'current', 'recent', 'this year', 'in 2023', 'person:', 'place:']
    dalle_keywords = ['draw', 'create', 'imagine']
    
    for keyword in wolfram_keywords:
        if re.search(keyword, query, re.IGNORECASE):
            return 'wolfram'
    
    for keyword in google_keywords:
        if keyword.lower() in query.lower():
            return 'google'
    for keyword in read_webpage_keywords:
        if re.search(url_pattern, query):
            return 'read_webpage'
    for keyword in dalle_keywords:
        if keyword.lower() in query.lower():
            return 'dalle'
    return 'openai'

if authentication_status:
    # Inside your Streamlit app
    number_of_images = st.sidebar.selectbox("Choose Number of Images",(1, 2, 3, 4))
    image_size = st.sidebar.selectbox("Choose Image Resolution",("1024x1024", "1024x1792", "1792x1024"))
    uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

    if uploaded_files is None or len(uploaded_files) == 0:
        st.sidebar.info("Upload files via the uploader in the sidebar.")
    else:
    
        # Loop through uploaded files
        for uploaded_file in uploaded_files:
            
            # Get the file extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            
            if file_extension == '.csv':
                result = handle_csv(uploaded_file)
            elif file_extension == '.txt':
                result = handle_txt(uploaded_file)
            elif file_extension == '.pdf':
                result = handle_pdf(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                result = handle_excel(uploaded_file, file_extension)
            elif file_extension.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif']:
                result = handle_image(uploaded_file)
            else:
                result = "Unsupported file type."
            
            st.write(result)

    # Check for new user input
    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        api_choice = get_api_choice(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            if api_choice == 'wolfram':
                # Stream the message that Wolfram Alpha is being queried
                full_response += "Querying Wolfram Alpha..."
                message_placeholder.markdown(full_response + "▌")

                # Initialize an empty string to hold the Wolfram Alpha result
                wolfram_result = ""
                
                # Get the Wolfram Alpha result
                wolfram_raw_result = wolframFunction(prompt)

                # Reset the full_response to empty so "Querying Wolfram Alpha..." disappears
                full_response = ""

                # Iterate through each character in the Wolfram Alpha result
                for char in wolfram_raw_result:
                    # Add each character to the full Wolfram Alpha result
                    wolfram_result += char

                    # Stream the current state of the full Wolfram Alpha result
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                
                # Remove the trailing cursor character after streaming is complete
                message_placeholder.markdown(full_response)
                
            # Inside the main logic where API is chosen
            elif api_choice == 'google':
                # Stream the message that Google Search is being performed
                full_response += "Performing Google Search..."
                message_placeholder.markdown(full_response + "▌")
                
                google_result = googleSearchFunction(prompt)  # Call the new Google Search function
                
                # Update the placeholder to indicate Google Search is complete
                full_response = "Google Search complete. Refining the answer..."
                message_placeholder.markdown(full_response + "▌")
                
                #refined_answer = get_refined_answer_from_openai(prompt, google_result, st.session_state.messages)
                
                # Stream the final refined answer
                full_response = google_result
                message_placeholder.markdown(full_response + "▌")
                
            elif api_choice == 'read_webpage':
                # Stream the message that the webpage is being read
                full_response += "Reading webpage content..."
                message_placeholder.markdown(full_response + "▌")
                
                webpage_content = read_webpage_content(re.search(url_pattern, prompt).group())

                # Update the placeholder to indicate the reading is complete and the assistant is refining the answer
                full_response = "Webpage read complete. Processing the information..."
                message_placeholder.markdown(full_response + "▌")
                
                refined_answer = get_refined_answer_from_openai(prompt, webpage_content, st.session_state.messages)
                
                # Stream the final refined answer
                full_response = refined_answer
                message_placeholder.markdown(full_response + "▌")
            
            elif api_choice == 'dalle':
                # Stream the message that the webpage is being read
                full_response += "Generating with Dalle..."
                message_placeholder.markdown(full_response + "▌")
                generation = generate_images(prompt)
                full_response = generation
                
                message_placeholder.markdown(full_response)
                
            else:
                # OpenAI API call
                for response in openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True
                ):
                    assistant_message = response.choices[0].delta.get("content", "")
                    full_response += assistant_message
                    message_placeholder.markdown(full_response + "▌")
                    
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

           
