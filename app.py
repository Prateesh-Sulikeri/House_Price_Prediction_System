from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

data = pd.read_csv("your_cleaned_data.csv") 
data = data.drop(['stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'], axis=1)
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)  

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    features = request.form.to_dict()  
    features = pd.DataFrame([list(features.values())], columns=list(features.keys()))
    prediction = model.predict(features)[0]  # Get prediction
    return render_template("prediction.html", prediction=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)