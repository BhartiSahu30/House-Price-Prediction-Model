import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# =====================================
# CUSTOM BACKGROUND
# =====================================

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-color:#0E1117;
color:white;
}

[data-testid="stSidebar"]{
background-color:#161A1D;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<h1 style='text-align:center;
color:white;
padding:20px;
border-radius:15px;
background:linear-gradient(to right,#141e30,#243b55);'>
🏠 AI House Price Prediction
</h1>
""", unsafe_allow_html=True)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("train.csv")

# =====================================
# DATA CLEANING
# =====================================

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# =====================================
# ENCODING
# =====================================

df_encoded = pd.get_dummies(df, drop_first=True)

# =====================================
# FEATURES & TARGET
# =====================================

X = df_encoded.drop("totalprice", axis=1)
y = df_encoded["totalprice"]

# =====================================
# LOAD MODEL
# =====================================

model = pickle.load(
    open("models/house_price_model.pkl", "rb")
)

# =====================================
# SIDEBAR
# =====================================

# =====================================
# SIDEBAR INPUTS
# =====================================

st.sidebar.header("🏡 House Features")

# BHK Slider
bhk = st.sidebar.slider(
    "Select BHK",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# Square Feet Slider
sqft = st.sidebar.slider(
    "Select Square Feet",
    min_value=500,
    max_value=10000,
    value=1500,
    step=100
)

# Location Dropdown
location = st.sidebar.selectbox(
    "Select Location",
    sorted(df["location"].unique())
)

# Property Type Dropdown
propertytype = st.sidebar.selectbox(
    "Select Property Type",
    sorted(df["propertytype"].unique())
)
st.sidebar.markdown("---")
# =====================================
# METRIC CARDS
# =====================================

col1, col2, col3 = st.columns(3)

col1.metric("BHK", bhk)

col2.metric("Square Feet", sqft)

col3.metric("Location", location)

# =====================================
# LOCATION PRICE MAP
# =====================================

location_price = {
    "Ahmedabad": 6000,
    "Mumbai": 25000,
    "Delhi": 18000,
    "Bangalore": 12000,
    "Chennai": 10000,
    "Hyderabad": 9000,
    "Pune": 11000,
    "Kolkata": 8000
}

avg_price_sqft = location_price.get(
    location,
    7000
)

st.info(
    f"Average Price Per Sqft in {location}: ₹ {avg_price_sqft}"
)

# =====================================
# CREATE INPUT DATA
# =====================================

input_dict = {
    "bhk": bhk,
    "sqft": sqft,
    "pricepersqft": avg_price_sqft
}

# Add remaining columns
for col in X.columns:
    if col not in input_dict:
        input_dict[col] = 0

# Encode property type
property_col = f"propertytype_{propertytype}"

if property_col in input_dict:
    input_dict[property_col] = 1

# Encode location
location_col = f"location_{location}"

if location_col in input_dict:
    input_dict[location_col] = 1

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Match column order
input_df = input_df[X.columns]

# =====================================
# PREDICTION BUTTON
# =====================================

if st.button("🔍 Predict House Price"):

    prediction = model.predict(input_df)[0]

    prediction = max(prediction, 0)

    # Prediction Card
    st.markdown(f"""
    <div style="
    background:linear-gradient(to right,#11998e,#38ef7d);
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:35px;
    font-weight:bold;
    margin-top:20px;">
    Estimated House Price <br><br>
    ₹ {prediction:,.0f}
    </div>
    """, unsafe_allow_html=True)

    st.success(
        f"Estimated House Price: ₹ {prediction:,.2f}"
    )
# =====================================
# TABS
# =====================================

tab1, tab2, tab3 = st.tabs([
    "📈 Analytics",
    "📊 Heatmap",
    "🗂 Dataset"
])

# =====================================
# TAB 1
# =====================================
with tab1:

    st.subheader("📊 Real-Time Price Distribution Analysis")

    # Filter dataset based on selected inputs
    filtered_df = df[
        (df["location"] == location) &
        (df["bhk"] == bhk)
    ]

    # If no exact match found
    if filtered_df.empty:

        filtered_df = df[
            (df["location"] == location)
        ]

    # Interactive Plotly Graph
    fig = px.bar(
        filtered_df.head(20),
        x="sqft",
        y="totalprice",
        color="totalprice",
        hover_data=[
            "bhk",
            "propertytype"
        ],
        title=f"House Prices in {location}",
        labels={
            "sqft": "Square Feet",
            "totalprice": "House Price (₹)"
        }
    )

    # Layout Styling
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Square Feet",
        yaxis_title="House Price (₹)",
        height=600
    )

    # Show Graph
    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TAB 3
# =====================================

with tab3:

    st.subheader("🗂 Dataset Preview")

    st.dataframe(df.head())
# =====================================
# FOOTER
# =====================================

st.markdown("""
<hr>
<center>
Made with ❤️ by Bharti Sahu
</center>
""", unsafe_allow_html=True)