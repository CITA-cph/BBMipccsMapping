from flask import Flask, request, jsonify

app = Flask(__name__)

latest_result = None

@app.route("/")
def home():
    return "Python server is running."

@app.route("/process", methods=["POST"])
def process():

    global latest_result

    data = request.json

    n1 = data["n1"]
    n2 = data["n2"]
    text = data["text"]
    tup = data["tup"]

    # Test computation
    latest_result = {
        "n1_squared": n1 ** 2,
        "n2_divided": n2 / 2,
        "text_twice": text + text,
        "tup0": tup[0],
        "tup1": tup[1]
    }

    return jsonify({"status": "ok"})


@app.route("/result", methods=["GET"])
def result():

    return jsonify(latest_result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)