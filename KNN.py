import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

churn_df = pd.read_csv('telecom_churn_clean.csv')

y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# New set of data. Use KNN classifier to predict the labels of the new data points.
X_new = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

y_pred = knn.predict(X_new)

print("Predictions: {}".format(y_pred))

Z = churn_df.drop("churn", axis=1).values

# Split into training and test sets
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.2, random_state=42, stratify=y)

# Fit the classifier to the training data
knn.fit(Z_train, y_train)

# Print the accuracy
print("Accuracy of KNN model: " + str(knn.score(Z_test, y_test)))

# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(Z_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(Z_train, y_train)
    test_accuracies[neighbor] = knn.score(Z_test, y_test)

print("Number of neighbors: " + str(neighbors))
print("Training accuracies: " + str(train_accuracies))
print("Test accuracies: " + str(test_accuracies))

# Plot training and test accuracies against the number of neighbours
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()
