import numpy as np
import pickle
import streamlit as st
from PIL import Image
import shap
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the saved model
loaded_model = pickle.load(open('final_model.sav', 'rb'))

# Function for Prediction
@st.cache_data(persist=True)
def predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType):
    input = np.array([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]])
    prediction = loaded_model.predict_proba(input)
    pred = '{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

# Function to generate SHAP values and plots
def generate_shap_values(model, input_data):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_data)
    return explainer, shap_values

# Helper function to display SHAP plots in Streamlit
def st_shap(plot, height=None):
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def main():
    st.set_page_config(
        page_title="Fraud Prediction App",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Set the background color to white and center the banner image
    st.markdown("""
        <style>
            body {
                background-color: white;
            }
            .main-header { background-color: #4CAF50; padding: 10px; text-align: center; }
            .main-title { color: white; }
            .sidebar .sidebar-content { background-color: #f0f0f0; padding: 10px; }
            .sidebar .sidebar-content h2 { color: #333; }
            .sidebar .sidebar-content .stNumberInput input, 
            .sidebar .sidebar-content .stRadio label, 
            .sidebar .sidebar-content .stSelectbox div { color: #333; }
            .banner-container { display: flex; justify-content: center; align-items: center; margin-top: 20px; }
            .banner-image { max-width: 100%; height: auto; }
        </style>
        <div class="main-header">
            <h1 class="main-title">Financial Transaction Fraud Prediction💰</h1>
        </div>
    """, unsafe_allow_html=True)


    # Tabs for Prediction and XAI Visualizations
    tab1, tab2 = st.tabs(["Prediction", "XAI Visualizations"])

    with tab1:
        st.sidebar.title("Financial Transaction Fraud Prediction System 🕵️")
        st.sidebar.subheader("Enter Transaction Details")

        TransactionAmt = st.sidebar.number_input("Transaction Amount (USD)", 0, 20000, step=1, help="Amount of the transaction in USD")
        card1 = st.sidebar.number_input("Payment Card 1 Amount (USD)", 0, 20000, step=1, help="First payment card amount")
        card2 = st.sidebar.number_input("Payment Card 2 Amount (USD)", 0, 20000, step=1, help="Second payment card amount")
        card4 = st.sidebar.radio("Payment Card Category", [1, 2, 3, 4], help="Type of payment card")
        st.sidebar.info("1 : Discover | 2 : Mastercard | 3 : American Express | 4 : Visa")
        card6 = st.sidebar.radio("Payment Card Type", [1, 2], help="Type of card usage")
        st.sidebar.info("1 : Credit | 2 : Debit")
        addr1 = st.sidebar.slider("Billing Zip Code", 0, 500, step=1, help="Zip code of billing address")
        addr2 = st.sidebar.slider("Billing Country Code", 0, 100, step=1, help="Country code of billing address")
        P_emaildomain = st.sidebar.selectbox("Purchaser Email Domain", [0, 1, 2, 3, 4], help="Email domain of the purchaser")
        st.sidebar.info("0 : Gmail (Google) | 1 : Outlook (Microsoft)  | 2 : Mail.com | 3 : Others | 4 : Yahoo")
        ProductCD = st.sidebar.selectbox("Product Code", [0, 1, 2, 3, 4], help="Code of the purchased product")
        st.sidebar.info("0 : C | 1 : H | 2 : R | 3 : S | 4 : W")
        DeviceType = st.sidebar.radio("Payment Device Type", [1, 2], help="Type of device used for payment")
        st.sidebar.info("1 : Mobile | 2 : Desktop")

        safe_html = """ 
        <img src="https://media.giphy.com/media/g9582DNuQppxC/giphy.gif" alt="confirmed" style="width:50%;height:auto;"> 
        """
        danger_html = """  
        <img src="https://media.giphy.com/media/8ymvg6pl1Lzy0/giphy.gif" alt="cancel" style="width:50%;height:auto;">
        """

        if st.button("Predict Fraud"):
            output = predict_fraud(card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType)
            final_output = output * 100
            st.subheader(f'Probability Score of Financial Transaction is {final_output:.2f}%')

            input_data = pd.DataFrame([[card1, card2, card4, card6, addr1, addr2, TransactionAmt, P_emaildomain, ProductCD, DeviceType]],
                                      columns=['card1', 'card2', 'card4', 'card6', 'addr1', 'addr2', 'TransactionAmt', 'P_emaildomain', 'ProductCD', 'DeviceType'])
            
            # Generate SHAP values
            explainer, shap_values = generate_shap_values(loaded_model, input_data)

            if final_output > 75.0:
                st.markdown(danger_html, unsafe_allow_html=True)
                st.error("**Alert! Financial Transaction is Fraudulent**")
            else:
                st.balloons()
                st.markdown(safe_html, unsafe_allow_html=True)
                st.success("**Success! Transaction is Legitimate**")

    with tab2:
        st.header("XAI Visualizations")

        if 'input_data' in locals():
            explainer, shap_values = generate_shap_values(loaded_model, input_data)
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("SHAP Summary Plot")
                fig_summary, ax_summary = plt.subplots(figsize=(6, 4))
                shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
                st.pyplot(fig_summary)

            with col2:
                st.subheader("SHAP Waterfall Plot")
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig_waterfall)

            st.subheader("SHAP Force Plot")
            shap.initjs()
            force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0], input_data.iloc[0])
            st_shap(force_plot, height=250)

        
        else:
            st.warning("No prediction data available. Please perform a prediction first.")

if __name__ == '__main__':
    main()
