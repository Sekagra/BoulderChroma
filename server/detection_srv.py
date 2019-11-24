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

def delete_subboxes(predictions):
    sort_predictions = sorted(predictions, key = lambda p: p["boundingBox"]["width"] * p["boundingBox"]["height"])

    delete_items = []
    for i, p in enumerate(sort_predictions):
        p_left =  p["boundingBox"]["left"]
        p_top =  p["boundingBox"]["top"]
        p_right =  p["boundingBox"]["width"] + p_left
        p_bottom =  p["boundingBox"]["height"] + p_top

        p_area = (
                max(0, p_right - p_left) *
                max(0, p_bottom - p_top)
            )

        for p2 in sort_predictions[(i + 1):]:
            p2_left =  p2["boundingBox"]["left"]
            p2_top =  p2["boundingBox"]["top"]
            p2_right =  p2["boundingBox"]["width"] + p2_left
            p2_bottom =  p2["boundingBox"]["height"] + p2_top

            inters_box_left = max(p_left, p2_left)
            inters_box_top = max(p_top, p2_top)
            inters_box_right = min(p_right, p2_right)
            inters_box_bottom = min(p_bottom, p2_bottom)

            inters_box_area = (
                max(0, inters_box_right - inters_box_left) *
                max(0, inters_box_bottom - inters_box_top)
            )

            if inters_box_area <= 0:
                continue

            if inters_box_area < .1 * p_area:
                continue

            delete_items.append(p)

            p2["boundingBox"]["left"] = min(p_left, p2_left)
            p2["boundingBox"]["top"] = min(p_top, p2_top)
            p2["boundingBox"]["width"] = max(p_right, p2_right) - p2["boundingBox"]["left"]
            p2["boundingBox"]["height"] = max(p_bottom, p2_bottom) - p2["boundingBox"]["top"]
                
            if p2["probability"] < p["probability"]:
                p2["tagId"] = p["tagId"]
                p2["tagName"] = p["tagName"]

    return [p for p in sort_predictions if p not in delete_items]


    


@app.route("/", methods=["post"])
def get_prediction():
    req_file = request.files['file']
    if req_file is None:
        print("Fuck you")
        return "Fuck you", 500

    img = Image.open(req_file)
    predictions = od_model.predict_image(img)
    return jsonify(delete_subboxes(predictions)), 200
    
    
if __name__ == "__main__":
    app.run(host="131.159.226.43",port=5000, debug=True)