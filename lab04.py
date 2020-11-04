import csv
import keras
import numpy
import os
import pandas
import sklearn
import time

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

IMAGE = []
CLASS = []
TAGS = []

# get current path
path = os.path.dirname(os.path.realpath(__file__))
DIR = f"{path}/fingerprints"


A = 1
L = 2
R = 3
T = 4
W = 5

def create_csv():
    """
    This function loops through the file adding all the tags
    into a csv file named "tags.csv".

    :return: None
    """
    with open("tags.csv", 'a') as csvFile:
        w = csv.writer(csvFile)
        header = ["History", "Gender"]
        w.writerow(header)

        for subdir, dirs, files in os.walk(DIR):
            for file in files:
                if file[-3:] == "txt":
                    with open(str(DIR) + "/" + str(file), 'r') as f:
                        data = []
                        for line in f:
                            s = line.split(": ")

                            if s[0] == "History":
                                h = s[1].split(" ")
                                data.append(h[1])
                            elif s[0] == "Gender":
                                data.append(s[1].strip("\n"))

                            if s[0] == "Class":
                                TAGS.append(s[1].strip("\n"))

                                if s[1].strip("\n") == "A":
                                    CLASS.append(A)
                                elif s[1].strip("\n") == "L":
                                    CLASS.append(L)
                                elif s[1].strip("\n") == "R":
                                    CLASS.append(R)
                                elif s[1].strip("\n") == "T":
                                    CLASS.append(T)
                                else:
                                    CLASS.append(W)

                        w.writerow(reversed(data))

    with open("labels.csv", 'a') as csvFile:
        w = csv.writer(csvFile)
        header = ["Class"]
        w.writerow(header)

        for element in TAGS:
            w.writerow([element])



def load_images():
    for subdir, dirs, files in os.walk(DIR):
        for file in files:
            if file[-3:] == "png":
                image = Image.open(str(DIR) + "/" + str(file))
                IMAGE.append(numpy.array(image))

def method1(x_train, x_test, y_train, y_test):
    start = time.time()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation='relu', input_shape=(512, 512,)))
    #model.add(keras.layers.Dense(512, activation='relu', input_dim=2))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer="adam", loss="mean_squared_error", metrics="accuracy")

    model.fit(numpy.array(x_train), numpy.array(y_train), batch_size=180, epochs=3, shuffle=True, verbose=1)
    # model.fit(numpy.array(x_train), numpy.array(y_train))
    result = []
    for i in x_test:
        result.append(model.predict(i))

    # print(numpy.array(x_test))
    # print("======")
    # print(numpy.array(y_test))
    # loss, accuracy, f1_score, precision, recall = model.evaluate(numpy.array(x_test), numpy.array(y_test), verbose=1)
    end = time.time()
    print("Method 1 took %d seconds" % (end - start))

def method2():
    start = time.time()

    data = pandas.read_csv("tags.csv")
    result = pandas.read_csv("labels.csv")

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(data.iloc[:, 0])
    data.iloc[:, 0] = encoder.transform(data.iloc[:, 0])

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(data.iloc[:, 1])
    data.iloc[:, 1] = encoder.transform(data.iloc[:, 1])

    data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.25)
    print("Finished Splitting")
    clf = RandomForestClassifier(random_state=25)
    print("Classier Completed")
    clf.fit(data_train, result_train)
    print("Fit Completed")
    predicted_array = clf.predict(data_test)
    print("Prediction Completed")

    print(str(sklearn.metrics.classification_report(result_test, predicted_array)))

    f = open("randomForest_result.txt", "w+")

    f.write(str(sklearn.metrics.confusion_matrix(result_test, predicted_array)))
    f.write(str(sklearn.metrics.classification_report(result_test, predicted_array)))

    f.close()

    end = time.time()
    print("Method 2 took %d seconds" % (end - start))

def method3():
    start = time.time()

    data = pandas.read_csv("tags.csv")
    result = pandas.read_csv("labels.csv")

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(data.iloc[:, 0])
    data.iloc[:, 0] = encoder.transform(data.iloc[:, 0])

    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(data.iloc[:, 1])
    data.iloc[:, 1] = encoder.transform(data.iloc[:, 1])

    data_train, data_test, result_train, result_test = train_test_split(data, result, test_size=0.25)
    print("Finished Splitting")
    scaler = sklearn.preprocessing.StandardScaler()
    print("Classier Completed")
    scaler.fit(data_train)

    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 256, 4), activation="relu", max_iter=250)
    mlp.fit(data_train, result_train.values.ravel())
    print("Fit Completed")

    predicted_array = mlp.predict(data_test)
    print("Prediction Completed")

    print(str(sklearn.metrics.classification_report(result_test, predicted_array)))

    f = open("mlp_result.txt", "w+")

    f.write(str(sklearn.metrics.confusion_matrix(result_test, predicted_array)))
    f.write(str(sklearn.metrics.classification_report(result_test, predicted_array)))
    f.close()

    end = time.time()
    print("Method 3 took %d seconds" % (end - start))

if __name__ == "__main__":
    create_csv()
    load_images()

    for image in IMAGE:
        sklearn.preprocessing.normalize(image)

    #(train, test) = train_test_split(SET, test_size=0.25)
    x_train, x_test, y_train, y_test = train_test_split(IMAGE, CLASS, test_size=0.25)

    method1(x_train, x_test, y_train, y_test)
    method2()
    method3()