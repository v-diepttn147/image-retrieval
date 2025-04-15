import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.preprocessing import OrdinalEncoder
from statistics import mean
from sklearn.model_selection import KFold   
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd
import sys

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  

def load_model(model_name):
    model = joblib.load('save_models/' + model_name + '.pkl')
    return model

# pipeline = load_model("full_pipeline")
# regressor = load_model("DecisionTreeRegressor")


def check_price_reasonability(area, rooms, toilets, asking_price, district, legal_status):
    pipeline = load_model("full_pipeline")
    regressor = load_model("DecisionTreeRegressor")

    # Input feature DataFrame
    X_input = pd.DataFrame([{
        "Area - M2": area,
        "Rooms": rooms,
        "Toilets": toilets,
        "District": district,           # required by pipeline
        "Legal status": legal_status      # required by pipeline
    }])

    # Preprocess
    X_processed = pipeline.transform(X_input)

    # Predict price
    predicted_price = regressor.predict(X_processed)[0]

    # Allow 20% margin (tune this)
    lower = predicted_price * 0.8
    upper = predicted_price * 1.2

    # print(f"\n🏠 Asking price: {asking_price:.1f} million VND")
    print(f"Predicted price: {predicted_price:.1f} million VND (acceptable range: {lower:.1f} - {upper:.1f})")

    return asking_price >= lower and asking_price <= upper
    #     print("⚠️ Price is outside the reasonable range!")
    #     return False
    # else:
    #     print("✅ Price is within the expected range.")
    #     return True

# check_price_reasonability(
#     image_path="./bed_485.jpg",
#     area=35,
#     rooms=1,
#     toilets=1,
#     asking_price=2000
# )

if __name__ == "__main__":

    area = sys.argv[1]
    rooms = sys.argv[2]
    toilets = sys.argv[3]
    asking_price = float(sys.argv[4])
    district = sys.argv[5]
    legal_status = sys.argv[6]
    result = check_price_reasonability(area, rooms, toilets, asking_price, district, legal_status)
    print(result)