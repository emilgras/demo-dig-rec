from datetime import datetime
from flask import Flask, render_template, request, jsonify

from utils import make_mnist
from sklearn.externals import joblib
from skimage import io as skio
from io import BytesIO
import base64

app = Flask(__name__)



@app.route('/')
@app.route('/home')
def home():
	return jsonify({'message': 'Digit Recognizer by Emil Gras'})

@app.route('/api/recognize', methods=['POST'])
def recognize():

	data = request.get_json(silent=True)['image']

	data = data[22:]

	img = skio.imread(BytesIO(base64.b64decode(data)))[:,:,3]

	img = make_mnist(img)

	clf = joblib.load('clf.pkl')

	number = clf.predict(img.reshape(1, -1))[0]

	return jsonify({'prediction': str(number)})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
