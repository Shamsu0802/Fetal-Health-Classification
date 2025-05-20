import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define labels for predicted values
health_labels = {
    1: "1-Normal",
    2: "2-Suspect",
    3: "3-Pathological"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        final_features = scaler.transform(final_features)
        prediction = model.predict(final_features)

        output_label = health_labels.get(prediction[0], "Unknown")
        return render_template('index.html', prediction_text='Predicted Fetal Health: {}'.format(output_label))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    try:
        data = request.get_json(force=True)
        features = [np.array(list(data.values()))]
        features = scaler.transform(features)
        prediction = model.predict(features)

        output_label = health_labels.get(prediction[0], "Unknown")
        return jsonify({"predicted_health": output_label})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

