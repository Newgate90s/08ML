from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree

# Importing the iris data set
iris = datasets.load_iris()

# Breaking the data set into two, one for training the other for testing
x = iris.data
y = iris.target

# x_train and y_train are used for training
# x_test and y_test is used for testing
# The data test, testing size is set to .5
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
# The first classifier we are using is the decision tree classifier from sklearn
my_decision_tree_classifier = tree.DecisionTreeClassifier()

print("Welcome to the Iris Data Set Classifying Program.")
print("\nHow it works:\nIt imports the iris data set from sklearn.")
print("\nData set explanation:\nThe data set is broken down into rows with 4 different values that apply to one specific flower.")
print("\nData set legend: Row[1st value = sepal lenght(cm), 2nd value = sepal width(cm), 3nd value = petal lenght(cm), 4th value = petal width(cm)")

# Using x_train and y_train to train the decision tree classifier
my_decision_tree_classifier.fit(x_train, y_train)

# Using .predict to predict our input of x_test
predictions_from_decision_tree_classifier = my_decision_tree_classifier.predict(x_test)

# Let's print out the predictions
# These correspond to the type of iris predicted
# for each row of our testing data

print("Here we print out our prediction.")
print("These correspond to the type of iris predicted.")
print("For each row of our testing data.")
print("")

# Now lets see how accurate our classifier was on the testing data
# To calculate our accuracy, we can compare our predicted labels with the true labels
# from sklearn.metrics import accuracy_score
print("The predictions from the decision tree classifier")
print(predictions_from_decision_tree_classifier)
print(accuracy_score(y_test, predictions_from_decision_tree_classifier))

############# implement the KNN ##################

from sklearn.neighbors import KNeighborsClassifier
my_k_nearest_neighbors_classifier = KNeighborsClassifier()

my_k_nearest_neighbors_classifier.fit(x_train, y_train)

predictions_from_KNeighborsClassifier = my_k_nearest_neighbors_classifier.predict(x_test)
print("The predictions from KNN classifier list")

print(accuracy_score(y_test, predictions_from_KNeighborsClassifier))


