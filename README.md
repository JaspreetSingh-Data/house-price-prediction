# 🏠 House Price Prediction

A machine learning web app that estimates house prices based on property features like area, bedrooms, bathrooms, location, and condition. Built with **scikit-learn** and deployed using **Streamlit**.

## Features

- Interactive Streamlit UI for entering property details
- Feature engineering pipeline (property age, area per room, location/condition interactions, etc.)
- Trained regression model served via a pre-fit `.pkl` pipeline
- Clean, two-column input form with instant price prediction

## Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Joblib (model serialization)

## Project Structure

```
house-price-prediction/
├── app.py                          # Streamlit application
├── house_price_model.pkl           # Trained model
├── final_feature_columns.pkl       # Feature column order used at inference
├── House_Price_Prediction_Dataset.csv  # Training dataset
├── notebooks/                      # EDA and model training notebooks
├── requirements.txt
└── README.md
```

## How It Works

1. User enters property details (area, bedrooms, bathrooms, floors, year built, location, condition, garage).
2. The app engineers additional features:
   - Property age
   - Total rooms
   - Area per bedroom / bathroom
   - Bedroom-to-bathroom ratio
   - Area × bedrooms interaction
   - Location/Condition/Garage combined categorical features
3. Features are reordered to match the training feature set.
4. The trained model predicts the estimated price.

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/JaspreetSingh-Data/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Notebooks

The `notebooks/` folder contains the exploratory data analysis and model training process used to build the final pipeline.

## Model Note

This model was built for learning/portfolio purposes. Predictions are estimates based on the training dataset and should not be used for real-world property valuations.

## Author

**Jaspreet Singh**
[GitHub](https://github.com/JaspreetSingh-Data) · [LinkedIn](https://www.linkedin.com/in/jaspreet-singh-54523a24b/) · [Kaggle](https://www.kaggle.com/dhiman_42)
