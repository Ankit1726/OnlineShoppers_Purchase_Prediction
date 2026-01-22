import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Online Shoppers Prediction",
    page_icon="🛒",
    layout="centered"
)

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>

.main {
    background-color: #f6f8fb;
}

.title-text {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #1f2937;
}

.subtitle-text {
    text-align: center;
    font-size: 16px;
    color: #4b5563;
    margin-bottom: 30px;
}

.block {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

.stButton button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
}

.stButton button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load model
# ---------------------------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------------------------
# Title
# ---------------------------------
st.markdown("<div class='title-text'>🛒 Online Shoppers Purchase Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Machine Learning model to predict customer buying intention</div>", unsafe_allow_html=True)

# ---------------------------------
# INPUTS
# ---------------------------------
st.markdown("<div class='block'>", unsafe_allow_html=True)
st.subheader("User Behavior Information")

col1, col2 = st.columns(2)

with col1:
    Administrative = st.number_input("Administrative Pages", 0)
    Informational = st.number_input("Informational Pages", 0)
    ProductRelated = st.number_input("Product Related Pages", 0)
    BounceRates = st.number_input("Bounce Rate", 0.0)
    PageValues = st.number_input("Page Value", 0.0)
    Weekend = st.selectbox("Weekend", [0, 1])

with col2:
    Administrative_Duration = st.number_input("Administrative Duration", 0.0)
    Informational_Duration = st.number_input("Informational Duration", 0.0)
    ProductRelated_Duration = st.number_input("Product Related Duration", 0.0)
    ExitRates = st.number_input("Exit Rate", 0.0)
    SpecialDay = st.number_input("Special Day", 0.0)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# CATEGORICAL
# ---------------------------------
st.markdown("<div class='block'>", unsafe_allow_html=True)
st.subheader("Technical & Visit Details")

col3, col4 = st.columns(2)

with col3:
    OperatingSystems = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
    Browser = st.selectbox("Browser", list(range(1,14)))
    Region = st.selectbox("Region", list(range(1,10)))

with col4:
    TrafficType = st.selectbox("Traffic Type", list(range(1,21)))
    Month = st.selectbox("Month",
        ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    )
    VisitorType = st.selectbox(
        "Visitor Type",
        ["Returning_Visitor","New_Visitor","Other"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# DATAFRAME
# ---------------------------------
input_df = pd.DataFrame([{
    "Administrative": Administrative,
    "Administrative_Duration": Administrative_Duration,
    "Informational": Informational,
    "Informational_Duration": Informational_Duration,
    "ProductRelated": ProductRelated,
    "ProductRelated_Duration": ProductRelated_Duration,
    "BounceRates": BounceRates,
    "ExitRates": ExitRates,
    "PageValues": PageValues,
    "SpecialDay": SpecialDay,
    "Weekend": Weekend,
    "OperatingSystems": OperatingSystems,
    "Browser": Browser,
    "Region": Region,
    "TrafficType": TrafficType,
    "Month": Month,
    "VisitorType": VisitorType
}])

# ---------------------------------
# PREDICTION
# ---------------------------------
if st.button("Predict Purchase"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to Purchase\n\nConfidence: {probability:.2%}")
    else:
        st.error(f"❌ Not Likely to Purchase\n\nConfidence: {1-probability:.2%}")
