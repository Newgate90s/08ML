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
print("\nHow it works:\nIt imports the iris data set from sklearn that has specific values that the classifiers use to learn how to distiguish the flower species.")
print("\nData set explanation:\nThe data set is broken down into two parts. "
      "\nPart one contain 4 values that belong to a specific flower"
      "\nPart two contains the flower label for those part one values.")


# Asking user for input to display part 21 of the two part training dataset
wait = input("\nPress enter to view first part of the data set used for training.")
print("="*80)
print("\nData set legend: \nRow [1st value = sepal lenght(cm), 2nd value = sepal width(cm), 3nd value = petal lenght(cm), 4th value = petal width(cm)\n")
print(x_train)
print("="*80)
# Asking user for input to display part 2 of the two part training dataset
wait = input("\nPress enter to view second part of the data set used for training.")
print("="*80)
print("\nData set legend:\n0 = Setosa, 1 = Versicolor, 2 = Virginica")
print(y_train)
print("="*80)

# Using x_train and y_train to train the decision tree classifier
my_decision_tree_classifier.fit(x_train, y_train)

# Using .predict to predict our input of x_test
predictions_from_decision_tree_classifier = my_decision_tree_classifier.predict(x_test)

# Asking user for input to display the dataset used for testing
print("\n\nNow we input a data set for our two classifiers to test their predictions.")
wait = input("\nPress enter to display the data set used for testing and input the data set")
print("="*80)
print("\nData set legend: \nRow [1st value = sepal lenght(cm), 2nd value = sepal width(cm), 3nd value = petal lenght(cm), 4th value = petal width(cm)\n")
print(x_test)
print("="*80)

# Asking user for input to print out the predictions and accuracy of our first classifier, the decision tree classifier from sklearn
wait = input("Press enter to view results of first classifier")
print("="*80)
print("Classifier 1 results:")
print(predictions_from_decision_tree_classifier)
print("\nClassifier 1 accuracy:")
print(accuracy_score(y_test, predictions_from_decision_tree_classifier))
print("="*80)


# Importing our second classifier, KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
my_k_nearest_neighbors_classifier = KNeighborsClassifier()

# Using x_train and y_train again to train our second classifier
my_k_nearest_neighbors_classifier.fit(x_train, y_train)

# Using the x_test input to do our predictions
predictions_from_KNeighborsClassifier = my_k_nearest_neighbors_classifier.predict(x_test)

# Asking user for input to pint out the predictions and accuracy of our second classifier, the k neighbors classifier from sklearn
wait = input("Press enter to view results of first classifier")
print("="*80)
print("Classifier 2 results:")
print(predictions_from_KNeighborsClassifier)
print("\nClassifier 2 accuracy:")
print(accuracy_score(y_test, predictions_from_KNeighborsClassifier))
print("="*80)


