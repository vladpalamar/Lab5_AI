import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Classify data using Ensem-ble Learning techniques")
    parser.add_argument('--classifier-type', dest='classifier_type', re - quired = True,
                                                                                   choices = ['rf',
                                                                                              'erf'], help = "Type of classifier to use; can be either 'rf' or 'erf'")
    return parser


args = build_arg_parser().parse_args()
classifier_type = args.classifier_type

input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
x, y = data[:, :-1], data[:, -1]

class_0 = np.array(x[y == 0])
class_1 = np.array(x[y == 1])
class_2 = np.array(x[y == 2])

plt.figure()

plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
            edgecol - ors = "black", linewidth = 1, marker = 's', label = 'Class 0')

plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecol - ors = "black", linewidth = 1, marker = 'o', label = 'Class 1')

plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
            edgecol - ors = 'black', linewidth = 1, marker = '^', label = 'Class 2')

plt.title('Incoming data')
plt.legend()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if classifier_type == 'rf':
    classifier = RandomForestClassifier(**params)
else:
    classifier = ExtraTreesClassifier(**params)

classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train)

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test)

class_names = ['Class-0', 'Class-1', 'Class-2']
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), tar - get_names = class_names))
print("#" * 40 + "\n")

print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

plt.show()
