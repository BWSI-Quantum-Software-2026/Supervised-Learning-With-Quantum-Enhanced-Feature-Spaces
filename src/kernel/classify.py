import numpy as np
from sklearn.svm import SVC

def train_svm(K_train, y_train, C=1.0):
    # "precomputed" tells sklearn not to compare the points itself because we already did that on the quantum side and we're handing over the matrix thats finished
    classifier = SVC(kernel="precomputed", C=C)

    # this checks where the boundary between +1 and -1 goes
    classifier.fit(K_train, y_train)
    return classifier


def predict_svm(classifier, K_test_train):
    # K_test_train holds each test point's similarity to every training point
    # and then sklearn uses those plus the boundary it learned to label each one +1 or -1 
    return classifier.predict(K_test_train)