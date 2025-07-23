import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder         #drawback increases dimension
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import mlflow.sklearn
import joblib

mlflow.sklearn.autolog()

df = pd.read_csv("airbnb_listings.csv")

x = df.drop(columns=["ListingID","PricePerNight"])
y = df["PricePerNight"]


#handling categorical variables using onehot encoder & column transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown='ignore'), ["City", "RoomType"])],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor",transformer),
    ("regressor",RandomForestRegressor(n_estimators=100,random_state=42))
])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

#here model will be save in mlflow page
with mlflow.start_run():
    model.fit(x_train,y_train)
    joblib.dump(model,"airbnb.pkl")
    print("model saved succesfully")