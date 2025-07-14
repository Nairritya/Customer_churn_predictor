from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Make prediction
        prediction = model.predict(final_features)

        # Interpret result
        output = 'Yes' if prediction[0] == 1 else 'No'
        return render_template('index.html', prediction_text=f"Churn Prediction: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
