# Import my data set
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree

iris = datasets.load_iris()

# I am going to partition my data into two sets
# 1/2 for testing and the other half for training
x = iris.data
y = iris.target

# Here I am using xtrain and ytrain for my training data
# and xtest and y test for my test data
#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# Now I am going to our go to classifier, decision tree
#from sklearn import tree
my_decision_tree_classifier = tree.DecisionTreeClassifier()

# Now I want to train the classifier on my training data
my_decision_tree_classifier.fit(x_train, y_train)

# Call predict method to classify our testing data
predictions_from_decision_tree_classifier = my_decision_tree_classifier.predict(x_test, y_test)

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
my_k_nearest_neighbors_classifier = KNeighborsClassifier

my_k_nearest_neighbors_classifier.fit(x_train, y_train)

predictions_from_KNeighborsClassifier = my_k_nearest_neighbors_classifier.predict(x_test)
print("The predictions from KNN classifier list")

print(accuracy_score(y_test, predictions_from_KNeighborsClassifier))


