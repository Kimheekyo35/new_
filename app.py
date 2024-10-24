from flask import Flask, request, jsonify
import joblib
from model import chatbot_message
import numpy as np
import json
#저장된 모델 불러오기
# MODEL_PATH="./model.pkl"
# loaded_model=joblib.load(MODEL_PATH)
def model(query):
	return chatbot_message(query)
app = Flask(__name__)

# 사용자가 "/"로 요청했을 시 실행되는 함수입니다.
# @app.route("/")
# def index():
# 	return "<p>Hello, World!</p>"

# @app.route("/hello")
# def hello_world():
	# return "<p>Hello, Users!</p>"

@app.route("/predict",methods=["POST","PUT"])
def inference():
	# my_set={'hello','world'}
	# my_list=list(my_set)
	# return json.dumps(my_list), 200
	#data=np.array(request.get_json()['data'])
	  # NumPy 배열을 리스트로 변환
	query = request.get_json()['query']
	chatbot_response=model(query)
	# chatbot_json=json.dumps(chatbot_response)

	return jsonify({'챗봇대답':chatbot_response})
	
if __name__ == "__main__":
	# 디버그 모드로 실행하며, 모든 IP에서의 접근을 허용합니다. 포트는 5002번을 사용합니다.
	app.run(debug=True, host='0.0.0.0', port=5002)