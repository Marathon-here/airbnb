# #implementing the application with flask and model from model.py
# from flask import Flask,render_template,request
# import joblib
# import pandas as pd

# app = Flask(__name__)

# #load the model from pickle file
# model = joblib.load("airbnb.pkl")


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


# Backend ---> Takes data from html and sends back to html
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("airbnb.pkl")

# Route for homepage — shows the input form
@app.route("/")
def index():
    return render_template("index.html")

# Route for handling form submission and prediction
@app.route("/predict", methods=["POST"])
def home():
    if request.method == "POST":
        # Get input values from form
        data = {
            "City": request.form.get("City"),
            "RoomType": request.form.get("RoomType"),
            "Bedrooms": int(request.form.get("Bedrooms")),
            "Bathrooms": int(request.form.get("Bathrooms")),
            "GuestsCapacity": int(request.form.get("GuestsCapacity")),
            "HasWifi": int(request.form.get("HasWifi")),
            "HasAC": int(request.form.get("HasAC")),
            "DistanceFromCityCenter": int(request.form.get("DistanceFromCityCenter"))
        }

        # Create a one-row DataFrame from the input
        df = pd.DataFrame([data])

        # Predict price using the model
        prediction = model.predict(df)[0]

        # Show the form again and display the prediction
        return render_template("index.html", predict=prediction)
    
    else:
        # If method is not POST, just show the form
        return render_template("index.html")

# Start the Flask development server
if __name__ == "__main__":
    app.run(debug=True)