# Build the House Price Prediction Application

# creating app

import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    
    model = joblib.load(
        'house_price_model.pkl'
    )
    
    feature_columns = joblib.load(
        'final_feature_columns.pkl'
    )
    
    return model, feature_columns


model, final_feature_columns = load_model()


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(input_df):
    
    df = input_df.copy()
    
    # Use the SAME year used during training
    CURRENT_YEAR = 2026
    
    # Property Age
    df['Property_Age'] = CURRENT_YEAR - df['YearBuilt']
    
    # Total Rooms
    df['Total_Rooms'] = (
        df['Bedrooms'] + df['Bathrooms']
    )
    
    # Area Per Bedroom
    df['AreaPerBedrooms'] = (
        df['Area'] /
        df['Bedrooms'].replace(0, 1)
    )
    
    # Area Per Bathroom
    df['AreaPerBathrooms'] = (
        df['Area'] /
        df['Bathrooms'].replace(0, 1)
    )
    
    # Bedroom Bathroom Ratio
    df['BedroomBathroomRatio'] = (
        df['Bedrooms'] /
        df['Bathrooms'].replace(0, 1)
    )
    
    # Area × Bedrooms
    df['AreaBedroomsInsteraction'] = (
        df['Area'] * df['Bedrooms']
    )
    
    # Location + Condition
    df['LocationCondition'] = (
        df['Location'].astype(str)
        + '_'
        + df['Condition'].astype(str)
    )
    
    # Location + Garage
    df['LocationGarage'] = (
        df['Location'].astype(str)
        + '_'
        + df['Garage'].astype(str)
    )
    
    # Condition + Garage
    df['ConditionGarage'] = (
        df['Condition'].astype(str)
        + '_'
        + df['Garage'].astype(str)
    )
    
    # Drop YearBuilt
    df = df.drop(
        columns=['YearBuilt']
    )
    
    return df


# ==========================================
# APPLICATION TITLE
# ==========================================

st.title("🏠 Advanced House Price Prediction")
st.write(
    "Enter the property details below to estimate the house price."
)

st.divider()


# ==========================================
# USER INPUT
# ==========================================

col1, col2 = st.columns(2)


with col1:
    
    area = st.number_input(
        "Area",
        min_value=1,
        value=2000
    )
    
    bedrooms = st.selectbox(
        "Bedrooms",
        [1, 2, 3, 4, 5]
    )
    
    bathrooms = st.selectbox(
        "Bathrooms",
        [1, 2, 3, 4]
    )
    
    floors = st.selectbox(
        "Floors",
        [1, 2, 3, 4]
    )


with col2:
    
    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2026,
        value=2010
    )
    
    location = st.selectbox(
        "Location",
        [
            "Downtown",
            "Suburban",
            "Urban",
            "Rural"
        ]
    )
    
    condition = st.selectbox(
        "Condition",
        [
            "Excellent",
            "Good",
            "Fair",
            "Poor"
        ]
    )
    
    garage = st.selectbox(
        "Garage",
        [
            "Yes",
            "No"
        ]
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()


if st.button(
    "Predict House Price",
    use_container_width=True
):
    
    # Create raw input DataFrame
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Floors': [floors],
        'YearBuilt': [year_built],
        'Location': [location],
        'Condition': [condition],
        'Garage': [garage]
    })
    
    
    # Feature Engineering
    input_engineered = create_features(
        input_data
    )
    
    
    # Ensure correct feature order
    input_engineered = input_engineered[
        final_feature_columns
    ]
    
    
    # Prediction
    prediction = model.predict(
        input_engineered
    )[0]
    
    
    # ==========================================
    # DISPLAY RESULT
    # ==========================================
    
    st.success("Prediction completed successfully!")
    
    st.metric(
        label="Estimated House Price",
        value=f"{prediction:,.2f}"
    )
    
    
    # Show entered property details
    st.subheader("Property Details")
    
    st.dataframe(
        input_data,
        use_container_width=True
    )
    
    
    # # Important model limitation
    # st.warning(
    #     "Model Note: This model was developed for "
    #     "educational purposes. During evaluation, the "
    #     "dataset showed weak predictive relationships "
    #     "between the available features and Price, so "
    #     "predictions should not be treated as accurate "
    #     "real-world valuations."
    # )