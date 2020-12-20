from flask import Flask, request, jsonify
from prediction import Keyword_Spotting_Service

app = Flask(__name__)

@app.route("/catch", methods=["POST"])
def catch():    
    if request.method == "POST":
        data = {"answer": "This from jsonify"}
        return jsonify(data)
    else:
        data = {"answer": "error post"}
        return jsonify(data)

@app.route("/get_predict", methods=["POST"])
def get_predict():
    if request.method == "POST":

        kss = Keyword_Spotting_Service()
        k1 = kss.predict("suaraAyah.wav")
        # print(f"Predicted keyword: {k1}")
        data = {"predicted": k1}
        return jsonify(data)

    else:
        data = {"predicted": "error when predicting"}
        return jsonify(data)
    

if __name__ == "__main__":
    app.run(debug=True)


