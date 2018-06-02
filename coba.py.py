import os
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import glob
import csv
# import os

# from pcd import *

app = Flask(__name__)


@app.route('/')
def index():
    return redirect('/static/index.html')

def grayscale(source):
    row, col, ch = source.shape
    graykanvas = np.zeros((row, col, 1), np.uint8)
    for i in range(0, row):
        for j in range(0, col):
            blue, green, red = source[i, j]
            gray = red * 0.299 + green * 0.587 + blue * 0.114
            graykanvas.itemset((i, j, 0), gray)
    return graykanvas

def substract(img, subtractor):
    grey = grayscale(img)
    row, col, ch = img.shape
    canvas = np.zeros((row, col, 3), np.uint8)
    for i in range (0, row):
        for j in range(0, col):
            b, g, r = img[i,j]
            subs = int(grey[i,j]) - int(subtractor[i,j])
            if(subs<0):
                canvas.itemset((i, j, 0), 0)
                canvas.itemset((i, j, 1), 0)
                canvas.itemset((i, j, 2), 0)
            else:
                canvas.itemset((i, j, 0), b)
                canvas.itemset((i, j, 1), g)
                canvas.itemset((i, j, 2), r)
    return canvas


@app.route('/mangga', methods=['POST'])
def mangga():

	# fixed-sizes for image
	fixed_size = tuple((100, 100))
	#filename = open('output/daun_data.csv', 'r')
	#dataframe = pandas.read_csv(filename)
	global_features = []

	image = cv2.imread('test.jpg')
	image = cv2.resize(image, fixed_size)
	count = 1
    data = []
    hsv = cv2.cvtColor(filename, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_RGB2GRAY
    ret,biner_threshold = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY 
    kernel3 = np.ones((9, 9), np.uint8)
    dilation3 = cv2.dilate(biner_threshold, kernel3, iterations=15)
    erotion3 = cv2.erode(dilation3, kernel3, iterations=15
    # cv2.imshow('gray', erotion3)
    # cv2.imshow('gray1', gray
    biner_threshold = cv2.bitwise_not(erotion3)
    final = substract(filename, biner_threshold)
    final1 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY
    hitam = 0
    hijau = 0
    berat = 0
    full  = 
    red = 0
    blue = 0
    green = 
    r_size = 0
    b_size = 0
    g_size = 
    #proposi RGB
    row, col = final1.shape
    for i in range(0, row):
        for j in range(0, col):
            val = final1[i,j]
            b, g, r = final[i,j]
            #print(b,g,r
            #if(g!=0 and r!=0):
            if (val!=0):
                #if(b>20): hijau = hijau + 1
                if(val>15 and val < 65): hijau=hijau+1
                else: hitam = hitam+
            red = red + r
            green = green + g
            blue = blue + 
            if(r): r_size = r_size + 1
            if(g): g_size = g_size + 1
            if(b): b_size = b_size + 
    hijau_final = float(hijau)/(hitam+hijau)
    hitam_final = float(hitam)/(hitam+hijau)
    r_final = float(red)/r_size
    g_final = float(green)/g_size
    b_final = float(blue)/b_siz
    berat = hitam+hijau
    full = row*col
    berat = float(berat)/ful

	global_feature = np.hstack([r_final, g_final, b_final, hijau_final, hitam_final, berat])
	prediction = clf.predict(global_feature.reshape(1,-1))[0]
	return render_template('index.html',prediksi = prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')

@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    app.run(debug=False, port=5000)