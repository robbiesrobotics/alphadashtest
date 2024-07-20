import json
import logging
import sys
from PIL import Image
import database as db
import streamlit_authenticator as stauth
import traceback
import openai
import pandas as pd
import streamlit as st
from plotly.graph_objs import Figure
from pydantic import BaseModel
from chat2plot import ResponseType, chat2plot
from chat2plot.chat2plot import Chat2Vega

sys.path.append("../../")

# Set up page configuration and favicon
basewidth = 600
favicon = Image.open("static/alpha.png")
wpercent = (basewidth / float(favicon.size[0]))
hsize = int((float(favicon.size[1]) * float(wpercent)))
img = favicon.resize((basewidth, hsize), Image.LANCZOS)
img.save('static/resized_image.png')
st.set_page_config(
    page_title="Chat Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Retrieve secrets
ZAPIER_CLIENT_SECRET = st.secrets["ZAPIER_CLIENT_SECRET"]
ZAPIER_CLIENT_ID = st.secrets["ZAPIER_CLIENT_ID"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# User Authentication Creation
users = db.fetch_all_users()
usernames = [user["key"] for user in users]
names = [user["name"] for user in users]
hashed_passwords = [user["password"] for user in users]
authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "AlphaChat", "abcdef", cookie_expiry_days=1)
name, authentication_status, username = authenticator.login("Login", "sidebar")

# Authentication Status Conditionals
if st.session_state["authentication_status"]:
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password')

# Initialize Logger
def initialize_logger():
    logger = logging.getLogger("root")
    handler = logging.StreamHandler(sys.stdout)
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]
    return True

if "logger" not in st.session_state:
    st.session_state["logger"] = initialize_logger()

# API Key setup
api_key = openai.api_key = OPENAI_API_KEY
if authentication_status:
    st.header("Alpha Chat Analysis")
    st.text("1. Upload a .csv file to begin")
    csv_file = st.sidebar.file_uploader("Upload CSV File", type={"csv"})
    if api_key and csv_file:
        df = pd.read_csv(csv_file)
        st.write("Tabulated Data:")
        st.write(df.head())

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        st.text(f'Chat History For {st.session_state["name"]}')

        def initialize_c2p():
            st.session_state["chat"] = chat2plot(
                df,
                st.session_state["chart_format"],
                verbose=True,
                description_strategy="head",
            )

        def reset_history():
            initialize_c2p()
            st.session_state["generated"] = []
            st.session_state["past"] = []

        with st.sidebar:
            chart_format = st.selectbox(
                "Chart format",
                ("simple", "vega"),
                key="chart_format",
                index=0,
                on_change=initialize_c2p,
            )

            #st.button("Reset conversation history", on_click=reset_history)
            #authenticator.logout('Logout', 'sidebar',) 
        if "chat" not in st.session_state:
            initialize_c2p()

        c2p = st.session_state["chat"]

        chat_container = st.container()
        input_container = st.container()
        #st.sidebar.write("[Terms of Service](https://docs.google.com/document/e/2PACX-1vRsnJ_liUiUnyrysB380Thgcu-jBRZ57YQgvXusDVO11F4QGe49sea5iYV1SJuaSKDbg9D6OhXDqPMr/pub)") 
        #st.sidebar.write("[Privacy Policy](https://docs.google.com/document/e/2PACX-1vRGFn8CTVLdRdjmNJ9DPusSmiwcjfxDKO9K8yh0cyR_Zazb0kLGqv3gEoRhKOIOWxkWTOpPtUWXyeFt/pub)")

        def submit(input_text):
            submit_text = input_text
            st.session_state["input"] = ""
            with st.spinner(text="Waiting for Alpha's response..."):
                try:
                    if isinstance(c2p, Chat2Vega):
                        res = c2p(submit_text, config_only=True)
                    else:
                        res = c2p(submit_text, config_only=False, show_plot=False)
                except Exception:
                    res = traceback.format_exc()
            st.session_state.past.append(submit_text)
            st.session_state.generated.append(res)

        def get_text():
            if input_text := st.chat_input(f'{st.session_state["name"]}"''s Prompt'):
                submit(input_text)
            return input_text

        with input_container:
            user_input = get_text()

        if st.session_state["generated"]:
            with chat_container:
                for i in range(len(st.session_state["generated"])):
                    st.chat_message("human").write(st.session_state["past"][i])

                    res = st.session_state["generated"][i]

                    if isinstance(res, str):
                        st.error(res.replace("\n", "\n\n"))
                    elif res.response_type == ResponseType.SUCCESS:
                        st.chat_message("ai").write(res.explanation)

                        col1, col2 = st.columns([2, 1])

                        with col2:
                            config = res.config
                            if isinstance(config, BaseModel):
                                st.code(
                                    config.json(indent=2, exclude_none=True),
                                    language="json",
                                )
                            else:
                                st.code(json.dumps(config, indent=2), language="json")
                        with col1:
                            if isinstance(res.figure, Figure):
                                st.plotly_chart(res.figure, use_container_width=True)
                            else:
                                st.vega_lite_chart(df, res.config, use_container_width=True)
                    else:
                        st.warning(
                            f"Failed to render chart. last message: {res.conversation_history[-1].content}",
                            icon="⚠️",
                        )
       # Logout button and clear chat button
    authenticator.logout('Logout', 'sidebar',)
    if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
        st.session_state.images = []

    # Links to terms of service and privacy policy
    st.sidebar.write("[Terms of Service](https://docs.google.com/document/d/e/2PACX-1vRsnJ_liUiUnyrysB380Thgcu-jBRZ57YQgvXusDVO11F4QGe49sea5iYV1SJuaSKDbg9D6OhXDqPMr/pub)")
    st.sidebar.write("[Privacy Policy](https://docs.google.com/document/d/e/2PACX-1vRGFn8CTVLdRdjmNJ9DPusSmiwcjfxDKO9K8yh0cyR_Zazb0kLGqv3gEoRhKOIOWxkWTOpPtUWXyeFt/pub)") 
       
else:
        st.write("# Welcome to Alpha Playground! 👋")

        #st.sidebar.success("Login and select the demo from above.")
        st.chat_message("ai").write("Hi. I'm Alpha, your friendly intelligent assistant. To get started, enter your username and password in the left sidebar.", avatar_style="avataaars-neutral", seed="Aneka114", key='intro_message_1')
        
        st.markdown(
            """
            Alpha Playground is a live demo of our app framework built specifically for
            Intelligent Systems and Business Solutions.
            
            👈 Login to see some examples
            of what Alpha can do!
            ### Want to learn more?
            - Check out [Calcifire Consulting](https://calcifireconsulting.com)
            - Jump into our [documentation](https://docs.streamlit.io)
            - Ask a question in our [community
                forums](https://discuss.streamlit.io)
            ### To See more complex demos Like:
            
            - Voice Chat (Speech to Text)
            - Voice Commands (Keyword Implementation)
            - Agent TTS (Text to Speech)
            - Computer Vision
            
            contact us at info@calcifireconsulting.com
        """
        )    