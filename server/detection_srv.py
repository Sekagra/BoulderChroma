from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import webcolors

from predict import TFLiteObjectDetection

MODEL_FILENAME = 'model.tflite'
LABELS_FILENAME = 'labels.txt'
prob_thres = .19

with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
od_model = TFLiteObjectDetection(MODEL_FILENAME, labels, prob_thres)

app = Flask(__name__)

COLOR = [
    ("yellow", u"#ffe9a6"),
    ("blue", u"#718bbe"),
    ("red", u"#df9288"),
    ("black", u"#4f4744"),
    ("green", u"#9abf94"),
    ("orange", u"#f2bfa4"),
    ("white", u"#d8d1c9"),
    ("blue", u"#395c44"),
    ("green", u"#31364a")
]

TAG_IDS = {
    "black" : 0, "blue" : 1, "green" : 2, "orange" : 3, "red" : 4, "white" : 5, "yellow" : 6
}

def find_nearest_color(img, prediction):
    w,h = img.size
    center_x = (prediction["boundingBox"]["left"] + 0.5 * prediction["boundingBox"]["width"]) * w
    center_y = (prediction["boundingBox"]["top"] + 0.5 * prediction["boundingBox"]["height"]) * h 
    print(center_x, center_y)
    r, g, b = img.getpixel((center_x, center_y))
    print("%x,%x,%x" % (r,g,b))
    return closest_colour((r,g,b))

def closest_colour(requested_colour):
    min_colours = {}
    
    for key, name in COLOR:
        r_c, g_c, b_c = webcolors.hex_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = key
    print(min_colours)
    return min_colours[min(min_colours.keys())]

@app.route("/", methods=["post"])
def get_prediction():
    req_file = request.files['file']
    if req_file is None:
        print("Fuck you")
        return "Fuck you", 500

    img = Image.open(req_file)
    predictions = od_model.predict_image(img)
    return jsonify(predictions), 200
    
    
if __name__ == "__main__":
    app.run(host="131.159.226.43",port=5000)