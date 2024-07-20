from alphaagent import query_agent, create_agent, write_response, decode_response
import pandas as pd
import streamlit as st
from PIL import Image
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import pygwalker as pyg
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import database as db

# Set up page configuration and favicon
basewidth = 600
favicon = Image.open("static/alpha.png")
wpercent = (basewidth / float(favicon.size[0]))
hsize = int((float(favicon.size[1]) * float(wpercent)))
img = favicon.resize((basewidth, hsize), Image.LANCZOS)
img.save('static/resized_image.png')
st.set_page_config(
    page_title="Alpha-Graph",
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

############### Authentication Status Conditionals ################
if st.session_state["authentication_status"]:
    st.write(f'Welcome *{st.session_state["name"]}*')
elif st.session_state["authentication_status"] is False:
    st.sidebar.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.sidebar.warning('Please enter your username and password') 
#####################################################################
# Initialize Session States
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-0613"
if "tab2_content" not in st.session_state:
    st.session_state.tab2_content = []
if "tab3_content" not in st.session_state:
    st.session_state.tab3_content = []
if "tab4_content" not in st.session_state:
    st.session_state.tab4_content = []

if authentication_status:
    data = st.sidebar.file_uploader("Upload a File", accept_multiple_files=True)
    ChatBot = st.container()
    
    with ChatBot:
        st.header("Visualize Data With Alpha")
        st.text("Intelligent chatbot with access to your documents. \nUpload a file to get insights")
        tab2, tab3, tab4 = st.tabs(["PDF Extract", "Data Profiler", "Insights Builder"])

    # PDF Extract to Dataframe
    with tab2:
        import streamlit as st
        import pytesseract
        from PyPDF2 import PdfReader
        from pdf2image import convert_from_path
        from PIL import Image
        import re
        import pandas as pd
        import plotly.express as px

        class Invoice:
            def __init__(self, invoice_number, invoice_date, due_date, payment_terms, service_data, invoice_total):
                self.invoice_number = invoice_number
                self.invoice_date = invoice_date
                self.due_date = due_date
                self.payment_terms = payment_terms
                self.service_data = service_data
                self.invoice_total = invoice_total

        invoice_numbers = []
        invoice_dates = []
        due_dates = []
        payment_terms_list = []
        service_names = []
        service_dates = []
        service_prices = []
        invoice_totals = []

        invoices = []

        uploaded_files = data

        if st.sidebar.button("Process Files"):
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    try:
                        file_extension = uploaded_file.name.split(".")[-1].lower()

                        if file_extension == "pdf":
                            pdf_reader = PdfReader(uploaded_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                        else:
                            img = Image.open(uploaded_file)
                            text = pytesseract.image_to_string(img)

                        invoice_number_pattern = r"Invoice no\.:\s*(\d+)"
                        invoice_date_pattern = r"Invoice date:\s*(\d{2}/\d{2}/\d{4})"
                        due_date_pattern = r"Due date:\s*(\d{2}/\d{2}/\d{4})"
                        payment_terms_pattern = r"Terms:\s*(.+)"
                        service_pattern = r"(\d+)\.(.*?)\s*\$(\d+\.\d{2})\s*Service date:\s*(\d{2}/\d{2}/\d{4})"
                        invoice_total_pattern = r"Total\s+\$\s*([\d.,]+)"

                        invoice_number_match = re.search(invoice_number_pattern, text)
                        invoice_date_match = re.search(invoice_date_pattern, text)
                        due_date_match = re.search(due_date_pattern, text)
                        payment_terms_match = re.search(payment_terms_pattern, text)
                        service_matches = re.findall(service_pattern, text)
                        invoice_total_match = re.search(invoice_total_pattern, text)

                        invoice_number = invoice_number_match.group(1) if invoice_number_match else None
                        invoice_date = invoice_date_match.group(1) if invoice_date_match else None
                        due_date = due_date_match.group(1) if due_date_match else None
                        payment_terms = payment_terms_match.group(1) if payment_terms_match else None
                        service_data = [(match[1].strip(), match[3], match[2]) for match in service_matches]
                        invoice_total = invoice_total_match.group(1) if invoice_total_match else None

                        invoice_numbers.append(invoice_number)
                        invoice_dates.append(invoice_date)
                        due_dates.append(due_date)
                        payment_terms_list.append(payment_terms)
                        service_names.extend([service_row[0] for service_row in service_data])
                        service_dates.extend([service_row[1] for service_row in service_data])
                        service_prices.extend([service_row[2] for service_row in service_data])
                        invoice_totals.append(invoice_total)

                        invoice = Invoice(invoice_number, invoice_date, due_date, payment_terms, service_data, invoice_total)
                        invoices.append(invoice)

                        st.text(text)

                    except Exception as e:
                        st.error(f"An error occurred while processing the file: {str(e)}")

            max_length = max(len(invoice_numbers), len(invoice_dates), len(due_dates), len(payment_terms_list), len(service_names), len(service_dates), len(service_prices), len(invoice_totals))

            invoice_numbers.extend([None] * (max_length - len(invoice_numbers)))
            invoice_dates.extend([None] * (max_length - len(invoice_dates)))
            due_dates.extend([None] * (max_length - len(due_dates)))
            payment_terms_list.extend([None] * (max_length - len(payment_terms_list)))
            service_names.extend([None] * (max_length - len(service_names)))
            service_dates.extend([None] * (max_length - len(service_dates)))
            service_prices.extend([None] * (max_length - len(service_prices)))
            invoice_totals.extend([None] * (max_length - len(invoice_totals)))

            invoice_data = pd.DataFrame(
                {
                    "Invoice Number": invoice_numbers,
                    "Invoice Date": invoice_dates,
                    "Due Date": due_dates,
                    "Payment Terms": payment_terms_list,
                    "Service Name": service_names,
                    "Service Date": service_dates,
                    "Service Price": service_prices,
                    "Invoice Total": invoice_totals,
                }
            )

            st.dataframe(invoice_data)
            st.header("Interactive Charts")

            x_axis = st.selectbox("Select X-axis data:", invoice_data.columns)
            y_axis = st.selectbox("Select Y-axis data:", invoice_data.columns)

            if x_axis and y_axis:
                fig = px.bar(invoice_data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                st.plotly_chart(fig)

    # Tab 3 - Data Profiler
    with tab3:
        st.title("Data Profiler")
        if st.sidebar.button("Profile Data"):
            if data is not None:
                for uploaded_file in data:
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    if file_extension == "pdf":
                        st.error("PDF files cannot be profiled. Please upload a CSV file.")
                    else:
                        df = pd.read_csv(uploaded_file)
                        pr = df.profile_report()
                        st.text("Profiling Data:")
                        st_profile_report(pr)
            else:
                st.warning("Please upload one or more files to profile.")


    # Tab 4 - Insights Builder (Advanced DataViz)
    with tab4:
        st.title("Insights Builder")
        if st.sidebar.button("Build Insights"):
            if data is not None:
                for uploaded_file in data:
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    if file_extension == "pdf":
                        st.error("PDF files cannot be used to build insights. Please upload a CSV file.")
                    else:
                        df = pd.read_csv(uploaded_file)
                        pyg_html = pyg.walk(df, return_html=True)
                        st.text("Insights:")
                        components.html(pyg_html, height=1000, scrolling=True)
            else:
                st.warning("Please upload one or more files to build insights.")

    authenticator.logout('Logout', 'sidebar',)
    st.sidebar.write("[Terms of Service](https://docs.google.com/document/d/e/2PACX-1vRsnJ_liUiUnyrysB380Thgcu-jBRZ57YQgvXusDVO11F4QGe49sea5iYV1SJuaSKDbg9D6OhXDqPMr/pub)")
    st.sidebar.write("[Privacy Policy](https://docs.google.com/document/d/e/2PACX-1vRGFn8CTVLdRdjmNJ9DPusSmiwcjfxDKO9K8yh0cyR_Zazb0kLGqv3gEoRhKOIOWxkWTOpPtUWXyeFt/pub)")
else:
    st.write("# Welcome to Alpha Playground! 👋")
    st.chat_message("ai").write("Hi. I'm Alpha, your friendly intelligent assistant. To get started, enter your username and password in the left sidebar.")
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
