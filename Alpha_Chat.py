####################         Imports      #######################################
import openai
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from alphaagent import generate_response
import streamlit_authenticator as stauth
import database as db
#############################################################################

########### Setup api Keys ###################################
openai.api_key = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
WOLFRAM_ALPHA_APPID = st.secrets["WOLFRAM_ALPHA_APPID"]
GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
###############################################################

st.set_page_config(
    page_title="Alpha Chat",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

####################### Create Memory ####################################
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
##########################################################################

if "messages" not in st.session_state:
    st.session_state.messages = []

##################### Reset Chat History Button ###########################
def reset_history():
    st.session_state.messages = []
###########################################################################

################### Setup LLM Chain ################################################
llm_chain = generate_response
#####################################################################################

####################  Main App #############################
def main():
    if authentication_status:
        ChatBot = st.container()
        with ChatBot:
            st.header("Chat With Alpha")
            st.text("An intelligent chatbot with access to the web. \nAsk Alpha questions or upload a file to get insights")
            st.text("Chat History:")

            # Display previous messages
            for msg in msgs.messages:
                if msg.type == "human":
                    st.chat_message("human").markdown(f'{name}: {msg.content}')
                    st.session_state.messages.append({"role": "user", "content": msg.content})
                elif msg.type == "ai":
                    st.chat_message("ai").markdown(f"Alpha: {msg.content}")
                    st.session_state.messages.append({"role": "ai", "content": msg.content})

            # User Input
            if prompt := st.chat_input():
                user_message = f'{name}: {prompt}'
                st.chat_message("human").markdown(user_message)

                # Generate and display AI response
                with st.spinner("Alpha is thinking..."):
                    response = llm_chain(prompt)
                    ai_message = "Alpha: " + response
                    st.chat_message("ai").markdown(ai_message)

        # Sidebar actions
        if st.sidebar.button("Logout"):
            authenticator.logout('Logout', 'sidebar',)
        st.sidebar.button("Clear Conversation", on_click=reset_history)
        st.sidebar.markdown("[Terms of Service](https://docs.google.com/document/d/e/2PACX-1vRsnJ_liUiUnyrysB380Thgcu-jBRZ57YQgvXusDVO11F4QGe49sea5iYV1SJuaSKDbg9D6OhXDqPMr/pub)") 
        st.sidebar.markdown("[Privacy Policy](https://docs.google.com/document/d/e/2PACX-1vRGFn8CTVLdRdjmNJ9DPusSmiwcjfxDKO9K8yh0cyR_Zazb0kLGqv3gEoRhKOIOWxkWTOpPtUWXyeFt/pub)") 
    ####################################################################################################

    else:
        st.write("Welcome to Alpha Playground! 👋")
        st.text("To get started, enter your username and password in the left sidebar.")

if __name__ == '__main__':
    main()
