import database as db
import streamlit_authenticator as stauth
import random
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Set up page configuration and favicon
basewidth = 600
favicon = Image.open("static/alpha.png")
wpercent = (basewidth / float(favicon.size[0]))
hsize = int((float(favicon.size[1]) * float(wpercent)))
img = favicon.resize((basewidth, hsize), Image.LANCZOS)
img.save('static/resized_image.png')
st.set_page_config(
    page_title="Alpha Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# USER Authentication Creation
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


#######################################
# PAGE SETUP
#######################################


if authentication_status:
    
    st.header("Alpha Auto Dashboard")
    with st.sidebar:
        st.header("Data Upload")
        uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True, type="xlsx")
        # Logout button and clear chat button
        authenticator.logout('Logout', 'sidebar',)
        if st.sidebar.button("Clear Conversation", key='clear_chat_button'):
            st.session_state.images = []

        # Links to terms of service and privacy policy
        st.sidebar.write("[Terms of Service](https://docs.google.com/document/d/e/2PACX-1vRsnJ_liUiUnyrysB380Thgcu-jBRZ57YQgvXusDVO11F4QGe49sea5iYV1SJuaSKDbg9D6OhXDqPMr/pub)")
        st.sidebar.write("[Privacy Policy](https://docs.google.com/document/d/e/2PACX-1vRGFn8CTVLdRdjmNJ9DPusSmiwcjfxDKO9K8yh0cyR_Zazb0kLGqv3gEoRhKOIOWxkWTOpPtUWXyeFt/pub)") 
        
    if uploaded_files is None:
        st.info(" Upload files via the uploader", icon="ℹ️")
        st.stop()

    #######################################
    # DATA LOADING
    #######################################
    if len(uploaded_files) == 0:
        st.info(" Upload Excel files via sidebar.")
        st.stop()

    @st.cache_data
    def load_data(files):
        df_list = [pd.read_excel(file) for file in files]
        df = pd.concat(df_list, ignore_index=True)
        return df

    df = load_data(uploaded_files)

    all_months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    with st.expander("Data Preview"):
        st.dataframe(
            df,
            column_config={"Year": st.column_config.NumberColumn(format="%d")},
        )

    #######################################
    # VISUALIZATION METHODS
    #######################################


    def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
        fig = go.Figure()

        fig.add_trace(
            go.Indicator(
                value=value,
                gauge={"axis": {"visible": False}},
                number={
                    "prefix": prefix,
                    "suffix": suffix,
                    "font.size": 28,
                },
                title={
                    "text": label,
                    "font": {"size": 24},
                },
            )
        )

        if show_graph:
            fig.add_trace(
                go.Scatter(
                    y=random.sample(range(0, 101), 30),
                    hoverinfo="skip",
                    fill="tozeroy",
                    fillcolor=color_graph,
                    line={
                        "color": color_graph,
                    },
                )
            )

        fig.update_xaxes(visible=False, fixedrange=True)
        fig.update_yaxes(visible=False, fixedrange=True)
        fig.update_layout(
            # paper_bgcolor="lightgrey",
            margin=dict(t=30, b=0),
            showlegend=False,
            plot_bgcolor="white",
            height=100,
        )

        st.plotly_chart(fig, use_container_width=True)


    def plot_gauge(
        indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
    ):
        fig = go.Figure(
            go.Indicator(
                value=indicator_number,
                mode="gauge+number",
                domain={"x": [0, 1], "y": [0, 1]},
                number={
                    "suffix": indicator_suffix,
                    "font.size": 26,
                },
                gauge={
                    "axis": {"range": [0, max_bound], "tickwidth": 1},
                    "bar": {"color": indicator_color},
                },
                title={
                    "text": indicator_title,
                    "font": {"size": 28},
                },
            )
        )
        fig.update_layout(
            # paper_bgcolor="lightgrey",
            height=200,
            margin=dict(l=10, r=10, t=50, b=10, pad=8),
        )
        st.plotly_chart(fig, use_container_width=True)


    def plot_top_right():
        sales_data = duckdb.sql(
            f"""
            WITH sales_data AS (
                UNPIVOT ( 
                    SELECT 
                        Scenario,
                        business_unit,
                        {','.join(all_months)} 
                        FROM df 
                        WHERE Year='2023' 
                        AND Account='Sales' 
                    ) 
                ON {','.join(all_months)}
                INTO
                    NAME month
                    VALUE sales
            ),

            aggregated_sales AS (
                SELECT
                    Scenario,
                    business_unit,
                    SUM(sales) AS sales
                FROM sales_data
                GROUP BY Scenario, business_unit
            )
            
            SELECT * FROM aggregated_sales
            """
        ).df()

        fig = px.bar(
            sales_data,
            x="business_unit",
            y="sales",
            color="Scenario",
            barmode="group",
            text_auto=".2s",
            title="Sales for Year 2023",
            height=400,
        )
        fig.update_traces(
            textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
        )
        st.plotly_chart(fig, use_container_width=True)


    def plot_bottom_left():
        sales_data = duckdb.sql(
            f"""
            WITH sales_data AS (
                SELECT 
                Scenario,{','.join(all_months)} 
                FROM df 
                WHERE Year='2023' 
                AND Account='Sales'
                AND business_unit='Software'
            )

            UNPIVOT sales_data 
            ON {','.join(all_months)}
            INTO
                NAME month
                VALUE sales
        """
        ).df()

        fig = px.line(
            sales_data,
            x="month",
            y="sales",
            color="Scenario",
            markers=True,
            text="sales",
            title="Monthly Budget vs Forecast 2023",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)


    def plot_bottom_right():
        sales_data = duckdb.sql(
            f"""
            WITH sales_data AS (
                UNPIVOT ( 
                    SELECT 
                        Account,Year,{','.join([f'ABS({month}) AS {month}' for month in all_months])}
                        FROM df 
                        WHERE Scenario='Actuals'
                        AND Account!='Sales'
                    ) 
                ON {','.join(all_months)}
                INTO
                    NAME year
                    VALUE sales
            ),

            aggregated_sales AS (
                SELECT
                    Account,
                    Year,
                    SUM(sales) AS sales
                FROM sales_data
                GROUP BY Account, Year
            )
            
            SELECT * FROM aggregated_sales
        """
        ).df()

        fig = px.bar(
            sales_data,
            x="Year",
            y="sales",
            color="Account",
            title="Actual Yearly Sales Per Account",
        )
        st.plotly_chart(fig, use_container_width=True)


    #######################################
    # STREAMLIT LAYOUT
    #######################################
    
    top_left_column, top_right_column = st.columns((2, 1))
    bottom_left_column, bottom_right_column = st.columns(2)
    
    with top_left_column:
        column_1, column_2, column_3, column_4 = st.columns(4)

        with column_1:
            plot_metric(
                "Total Accounts Receivable",
                6621280,
                prefix="$",
                suffix="",
                show_graph=True,
                color_graph="rgba(0, 104, 201, 0.2)",
            )
            plot_gauge(1.86, "#0068C9", "%", "Current Ratio", 3)

        with column_2:
            plot_metric(
                "Total Accounts Payable",
                1630270,
                prefix="$",
                suffix="",
                show_graph=True,
                color_graph="rgba(255, 43, 43, 0.2)",
            )
            plot_gauge(10, "#FF8700", " days", "In Stock", 31)

        with column_3:
            plot_metric("Equity Ratio", 75.38, prefix="", suffix=" %", show_graph=False)
            plot_gauge(7, "#FF2B2B", " days", "Out Stock", 31)
            
        with column_4:
            plot_metric("Debt Equity", 1.10, prefix="", suffix=" %", show_graph=False)
            plot_gauge(28, "#29B09D", " days", "Delay", 31)

    with top_right_column:
        plot_top_right()

    with bottom_left_column:
        plot_bottom_left()

    with bottom_right_column:
        plot_bottom_right()
    
        
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