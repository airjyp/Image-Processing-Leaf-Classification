from flask import Flask, redirect, request, jsonify,render_template
from PIL import Image
import io
import h5py
import numpy as np
import os
import cv2
import mahotas
import glob
import csv
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)


@app.route('/')
def index():
    return redirect('/static/index.html')

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def fd_histogram(image, mask=None):
    bins = 8
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


@app.route('/daun', methods=['POST'])
def daun():

	# fixed-sizes for image
	fixed_size = tuple((500, 500))


	filename = open('output/daun_data.csv', 'r')
	dataframe = pandas.read_csv(filename)

	kelas = dataframe.drop(dataframe.columns[:-1], axis=1)
	data = dataframe.drop(dataframe.columns[-1:], axis=1)

	# print(data)
	# print(kelas)

	# empty lists to hold feature vectors and labels
	global_features = []
	labels = []

	i, j = 0, 0
	k = 0
	num = 102

	# create all the machine learning models
	models = []
	models.append(('DECISION TREE ACCURACY', DecisionTreeClassifier(random_state=num)))

	# variables to hold the results and names
	results = []
	names = []
	scoring = "accuracy"

	# filter all the warnings
	import warnings
	warnings.filterwarnings('ignore')

	# 10-fold cross validation
	for name, model in models:
	    kfold = KFold(n_splits=10, random_state=7)
	    cv_results = cross_val_score(model, data, kelas, cv=kfold, scoring=scoring)
	    results.append(cv_results)
	    names.append(name)
	    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	    print(msg)


	import matplotlib.pyplot as plt

	# create the model - Random Forests
	clf  =  DecisionTreeClassifier(random_state=num)
	# fit the training data to the model
	clf.fit(data,kelas)

	image = cv2.imread('test.jpg')
	image = cv2.resize(image, fixed_size)
	fv_hu_moments = fd_hu_moments(image)
	fv_haralick   = fd_haralick(image)
	fv_histogram  = fd_histogram(image)
	global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
	prediction = clf.predict(global_feature.reshape(1,-1))[0]
	return render_template('result.html',prediksi = prediction)

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
