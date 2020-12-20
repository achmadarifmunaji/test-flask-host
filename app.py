from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/catch", methods=["POST"])
def catch():    
    if request.method == "POST":
        data = {"answer": "This from jsonify"}
        return jsonify(data)
    else:
        data = {"answer": "error post"}
        return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)


