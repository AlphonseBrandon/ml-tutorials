import preprocessing
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        predicted_price = preprocessing.data_pipeline(json_data)
        response = {'predicted_price': predicted_price}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) 