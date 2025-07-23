# #implementing the application with flask and model from model.py
# from flask import Flask,render_template,request
# import joblib
# import pandas as pd

# app = Flask(__name__)

# #load the model from pickle file
# import os
# model_path = os.path.join(os.path.dirname(__file__), "airbnb.pkl")
# model = joblib.load(model_path)


# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/check",methods = ["POST"])
# def home():
#     #request to post method ,to let user enter the details or input
#     if request.method == "POST":
#         data ={
#             "City":request.form.get("City"),
#             "RoomType":request.form.get("RoomType"),
#             "Bedrooms":int(request.form.get("Bedrooms")),
#             "Bathrooms":int(request.form.get("Bathrooms")),
#             "GuestsCapacity":int(request.form.get("GuestsCapacity")),
#             "HasWifi":int(request.form.get("HasWifi")),
#             "HasAC":int(request.form.get("HasAC")),
#             "Distance": int(request.form.get("DistanceFromCityCenter", 0)),
            
#         }
#         df = pd.DataFrame([data])   
#         print("Received Data:", data)         #converting to dataframe to make the model to  make prediction with all features at a time
#         prediction = model.predict(df)[0] #predicting the outcome parellelly with all features
#         return render_template("index.html",predictt = prediction) #sending the output to userinterface
#     else:
#         return render_template("index.html")

# if __name__=="__main__":
#     app.run()

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "airbnb.pkl")
model = joblib.load(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form.get("City")
        room_type = request.form.get("RoomType")
        bedrooms = int(request.form.get("Bedrooms"))
        bathrooms = int(request.form.get("Bathrooms"))
        guests = int(request.form.get("GuestsCapacity"))
        wifi = int(request.form.get("HasWifi"))
        ac = int(request.form.get("HasAC"))
        distance = float(request.form.get("DistanceFromCityCenter"))

        data = pd.DataFrame([[city, room_type, bedrooms, bathrooms, guests, wifi, ac, distance]],
                            columns=['City', 'RoomType', 'Bedrooms', 'Bathrooms', 'GuestsCapacity', 'HasWifi', 'HasAC', 'DistanceFromCityCenter'])

        prediction = model.predict(data)[0]

        user_data = {
            "City": city,
            "RoomType": room_type,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "GuestsCapacity": guests,
            "HasWifi": wifi,
            "HasAC": ac,
            "DistanceFromCityCenter": distance
        }

        return render_template("index.html", predict=prediction, user_data=user_data)

    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)

