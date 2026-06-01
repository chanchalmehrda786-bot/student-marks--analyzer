

from flask import Flask, request, jsonify
import numpy as np
import joblib

model = joblib.load('iris_model.pkl')

app = Flask(__name__)

# class mapping
classes = ["setosa", "versicolor", "virginica"]

@app.route("/")
def home():
    return "Flask API is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        sl = float(data['sl'])
        sw = float(data['sw'])
        pl = float(data['pl'])
        pw = float(data['pw'])

        user_data = np.array([[sl, sw, pl, pw]])

        prediction = model.predict(user_data)[0]

        # convert 0/1/2 → flower name
        result = classes[prediction]

        return jsonify({
            "prediction": int(prediction),
            "class_name": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 
