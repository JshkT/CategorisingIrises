# Classification of Iris flowers
# Simple project to better understand machine learning.
# Joshua Tan

import versionChecker


def import_libraries():
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

    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    data = read_csv(url, names=names)
    return data


def process_data(dataset):
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('class').size())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    versionChecker.check_version()
    dataset = import_libraries()
    process_data(dataset)

    print("End process")
