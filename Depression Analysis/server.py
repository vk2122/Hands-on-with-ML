from flask import Flask, request, render_template
from models import Model
#from facecv import run
import os
from cv2 import cv2 as cv
from PIL import ImageDraw
#run()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def faces_from_pil_image(pil_image):
    "Return a list of (x,y,h,w) tuples for faces detected in the PIL image"
    storage = cv.CreateMemStorage(0)
    facial_features = cv.Load('haarcascade_frontalface_alt.xml', storage=storage)
    cv_im = cv.CreateImageHeader(pil_image.size, cv.IPL_DEPTH_8U, 3)
    cv.SetData(cv_im, pil_image.tostring())
    faces = cv.HaarDetectObjects(cv_im, facial_features, storage)
    # faces includes a `neighbors` field that we aren't going to use here
    return [f[0] for f in faces]

def draw_faces(image_, faces):
    "Draw a rectangle around each face discovered"
    image = image_.copy()
    drawable = ImageDraw.Draw(image)

    for x, y, w, h in faces:
        absolute_coords = (x, y, x + w, y + h)
        drawable.rectangle(absolute_coords)

    return image

@app.route("/uploadImage",methods=["POST"])
def uploadImage():
    ig = request.files["image"]
    # cv code to recognize face and crop the image to get only the face
    ig.save("imgs\\"+ig.filename)

@app.route('/predict', methods=["POST"])
def predict():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    q10 = int(request.form['a10'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    model = Model()
    classifier = model.svm_classifier()
    prediction = classifier.predict([values])
    if prediction[0] == 0:
        result1 = 'Your Test result : No Depression'
    if prediction[0] == 1:
        result1 = 'Your Test result : Mild Depression'
    if prediction[0] == 2:
        result1 = 'Your Test result : Moderate Depression'
    if prediction[0] == 3:
        result1 = 'Your Test result : Moderately severe Depression'
    if prediction[0] == 4:
        result1 = 'Your Test result : Severe Depression'
    return render_template("result.html", result=result1)


app.secret_key = os.urandom(12)
app.run(port=5987, host='0.0.0.0', debug=True)