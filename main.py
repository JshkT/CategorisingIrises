# Classification of Iris flowers
# Simple project to better understand machine learning.
# Joshua Tan

# Requires SciPy
# I used the following line in terminal to install.
# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
# If it does not work for you, follow the link.
# https://www.scipy.org/install.html

import versionChecker

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def import_data():
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    data = read_csv(url, names=names)
    return data


def display_data(dataset):
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('class').size())

    # Univariate Plots
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2),
                 sharex=False, sharey=False)
    pyplot.show()

    # histograms
    dataset.hist()
    pyplot.show()

    # Multivariate Plots
    # scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()


def evaluate_models(dataset, X_train, Y_train):
    # Spot check algorithms
    # mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.
    #   Logistic Regression (LR)
    #   Linear Discriminant Analysis (LDA)
    #   K-Nearest Neighbors (KNN).
    #   Classification and Regression Trees (CART).
    #   Gaussian Naive Bayes (NB).
    #   Support Vector Machines (SVM).
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Compare Algorithms
    pyplot.boxplot(results, labels=names)
    pyplot.title('Algorithm Comparison')
    pyplot.show()


def get_predictions(model, X_train, Y_train, X_validation):
    model.fit(X_train, Y_train)
    return model.predict(X_validation)

# Evaluate predictions
def evaluate_predictions(predictions, Y_validation):
    print('Accuracy Score: %f' % accuracy_score(Y_validation, predictions))
    print('Confusion Matrix: \n%s' % confusion_matrix(Y_validation, predictions))
    print('Classification Report: \n%s' % classification_report(Y_validation, predictions))


# Press the green button in the gutter to run the script.
def main():
    # versionChecker.check_version()
    dataset = import_data()
    # display_data(dataset)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, y, test_size=0.20, random_state=1)

    # evaluate_models(dataset, X_train, Y_train)

    model = SVC(gamma='auto')
    predictions = get_predictions(model, X_train, Y_train, X_validation)

    evaluate_predictions(predictions, Y_validation)

    print("End process")


main()
