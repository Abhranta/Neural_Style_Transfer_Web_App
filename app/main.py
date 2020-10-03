from flask import Flask , request , jsonify, render_template

app = Flask(__name__)

@app.route("/")
def produce():
    #1 load the original image
    #2 load the style image
    #3 conver original to tensor
    #4 convert style to tensor
    #5 style transfer
    #6 give output
    return render_template("index.html")
