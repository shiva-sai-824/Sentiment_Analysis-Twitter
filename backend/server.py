from flask import Flask, request, jsonify
from flask_cors import CORS
from sentiment import predict

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

def analyze_sentiment(text):
    sentiment = predict(text)
    return sentiment

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    sentiment = analyze_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
