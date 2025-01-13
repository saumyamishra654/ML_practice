import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from data_preprocessing import preprocess
filepath = "/Users/saumyamishra/Desktop/Ashoka/sem 5/Introduction to Machine Learning/Saumya_Mishra_A1/decision_tree_task/data/train_data.csv"
dataset = preprocess(filepath)
dataset.head()

# Node class represents a node in the decision tree with split information and children nodes
class Node():

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None.
            gain: The gain of the split. Defaults to None.
            value: If this node is a leaf node, this attribute represents the predicted value
                for the target variable. Defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

# DecisionTree class contains methods for building and evaluating the decision tree model
class DecisionTree():
    """
    A decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
        """
        Splits the given dataset into two datasets based on the given feature and threshold.

        Parameters:
            dataset (ndarray): Input dataset.
            feature (int): Index of the feature to be split on.
            threshold (float): Threshold value to split the feature on.

        Returns:
            left_dataset (ndarray): Subset of the dataset with values less than or equal to the threshold.
            right_dataset (ndarray): Subset of the dataset with values greater than the threshold.
        """
        # Create empty arrays to store the left and right datasets
        left_dataset = []
        right_dataset = []
        
        # Loop over each row in the dataset and split based on the given feature and threshold
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        # Convert the left and right datasets to numpy arrays and return
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        Computes the entropy of the given label values.

        Parameters:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """
        entropy = 0

        # find the unique label values in y and loop over each value
        labels = np.unique(y)
        for label in labels:
            # find the examples in y that have the current label
            label_examples = y[y == label]
            # calculate the ratio of the current label in y
            pl = len(label_examples) / len(y)
            # calculate the entropy using the current label and ratio
            entropy += -pl * np.log2(pl)

        # final entropy value
        return entropy

    def information_gain(self, parent, left, right):
        """
        Computes the information gain from splitting the parent dataset into two datasets.

        Parameters:
            parent (ndarray): Input parent dataset.
            left (ndarray): Subset of the parent dataset after split on a feature.
            right (ndarray): Subset of the parent dataset after split on a feature.

        Returns:
            information_gain (float): Information gain of the split.
        """
        # set initial information gain to 0
        information_gain = 0
        # compute entropy for parent
        parent_entropy = self.entropy(parent)
        # calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        # compute entropy for left and right nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # calculate weighted entropy 
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        # calculate information gain 
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the given dataset.

        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain, 
              left and right datasets.
        """
        # dictionary to store the best split values
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        # loop over all the features
        for feature_index in range(num_features):
            #get the feature at the current feature_index
            feature_values = dataset[:, feature_index]
            #get unique values of that feature
            thresholds = np.unique(feature_values)
            # loop over all values of the feature
            for threshold in thresholds:
                # get left and right datasets
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                # check if either datasets is empty
                if len(left_dataset) and len(right_dataset):
                    # get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    # compute information gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split

    
    def calculate_leaf_value(self, y):
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        y = list(y)
        #get the highest present class in the array
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        # split the dataset into X, y values
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Get the best split
            best_split = self.best_split(dataset, n_samples, n_features)
            # Check if gain isn't zero
            if best_split["gain"]:
                # continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # compute leaf node value
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node value
        return Node(value=leaf_value)
    
    def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((X, y), axis=1)  
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        np.array(predictions)
        return predictions
    
    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector.

        Args:
        x (ndarray): The feature vector to predict the target value for.
        node (Node): The current node being evaluated.

        Returns:
        The predicted target value for the given feature vector.
        """
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

def confuse(y_true, y_pred):
    # calculate true negatives, false positives, false negatives, true positives
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TP = np.sum((y_pred == 1) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])

def accurate(y_true, y_pred):
    # calculate accuracy
    return np.mean(y_true == y_pred)

def precise(y_true, y_pred):
    # calculate precision
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall_score(y_true, y_pred):
    # calculate recall
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(y_true, y_pred):
    # calculate f1 score
    precision = precise(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_metrics(y_true, y_pred):
    # compute confusion matrix, accuracy, precision, recall, f1 score
    conf_matrix = confuse(y_true, y_pred)
    accuracy = accurate(y_true, y_pred)
    precision = precise(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

def save_metrics(metrics_output_path, accuracy, precision, recall, f1, conf_matrix):
    # save calculated metrics to a file
    with open(metrics_output_path, 'w') as metrics_file:
        metrics_file.write(f"Classification Metrics:\n")
        metrics_file.write(f"Accuracy: {accuracy:.4f}\n")
        metrics_file.write(f"Precision: {precision:.4f}\n")
        metrics_file.write(f"Recall: {recall:.4f}\n")
        metrics_file.write(f"F1-Score: {f1:.4f}\n")
        metrics_file.write(f"Confusion Matrix:\n")
        metrics_file.write(f"[[{conf_matrix[0, 0]}, {conf_matrix[0, 1]}],\n")
        metrics_file.write(f" [{conf_matrix[1, 0]}, {conf_matrix[1, 1]}]]\n")

def undersample(df, target_column, ratio=1):
    # Identify the minority and majority classes
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    # Separate the minority and majority classes
    df_minority = df[df[target_column] == minority_class]
    df_majority = df[df[target_column] == majority_class]
    
    # Determine the number of majority samples to keep (based on the 1:5 ratio)
    majority_sample_size = len(df_minority) * ratio
    
    # Randomly undersample the majority class to achieve the 1:5 ratio
    df_majority_undersampled = df_majority.sample(majority_sample_size, random_state=42)
    
    # Combine the undersampled majority class with the minority class
    df_undersampled = pd.concat([df_minority, df_majority_undersampled])
    
    # Shuffle the resulting DataFrame
    df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_undersampled

undersampled_df = undersample(dataset, 'isFraud', ratio=1)
undersampled_df.shape[0]
y = undersampled_df.pop('isFraud')
X = undersampled_df.copy()
def balanced_accuracy(y_true, y_pred):
    """Calculate the balanced accuracy for a multi-class classification problem.

    Parameters
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns
    -------
        balanced_acc : The balanced accuracy of the model
        TP: List of true positives for each class
        TN: List of true negatives for each class
        FP: List of false positives for each class
        FN: List of false negatives for each class
        
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Get the number of classes
    n_classes = len(np.unique(y_true))

    #  arrays to store TP, TN, FP, FN for each class
    TP = []
    TN = []
    FP = []
    FN = []

    #  arrays to store sensitivity and specificity for each class
    sen = []
    spec = []

    for i in range(n_classes):
        mask_true = y_true == i
        mask_pred = y_pred == i

        TP_i = np.sum(mask_true & mask_pred)
        TN_i = np.sum((mask_true == False) & (mask_pred == False))
        FP_i = np.sum((mask_true == False) & mask_pred)
        FN_i = np.sum(mask_true & (mask_pred == False))

        TP.append(TP_i)
        TN.append(TN_i)
        FP.append(FP_i)
        FN.append(FN_i)

        sensitivity = TP_i / (TP_i + FN_i) if (TP_i + FN_i) > 0 else 0
        specificity = TN_i / (TN_i + FP_i) if (TN_i + FP_i) > 0 else 0

        sen.append(sensitivity)
        spec.append(specificity)

    average_sen = np.mean(sen)
    average_spec = np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc, TP, TN, FP, FN

def accuracy(y_true, y_pred):
    """
    Computes the accuracy of a classification model.

    Parameters:
    ----------
        y_true (numpy array): A numpy array of true labels for each data point.
        y_pred (numpy array): A numpy array of predicted labels for each data point.

    Returns:
    ----------
        float: The accuracy of the model
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 
X = undersampled_df.copy()

shuffled_indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2)
test_indices = shuffled_indices[:test_size]
train_indices = shuffled_indices[test_size:]

X_train, X_test = X.iloc[train_indices].to_numpy(), X.iloc[test_indices].to_numpy()
y_train, y_test = y.iloc[train_indices].to_numpy().reshape(-1,1), y.iloc[test_indices].to_numpy().reshape(-1,1)
model = DecisionTree(2, 2)

# Fit the decision tree model to the training data.
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data.
predictions = model.predict(X_test)

# Calculate evaluating metrics
print(f"Model's Accuracy: {accuracy(y_test, predictions)}")
      
balanced_acc, TP, TN, FP, FN  = balanced_accuracy(y_test, predictions)
def compute_confusion_matrix(TP, TN, FP, FN):
    """Construct confusion matrix using TP, TN, FP, FN."""
    n_classes = len(TP)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    # Fill the confusion matrix
    for i in range(n_classes):
        confusion_matrix[i, i] = TP[i]  # True Positives for each class
        for j in range(n_classes):
            if i != j:
                confusion_matrix[i, j] = FP[j] if i == 1 else FN[i]  # False Positives and False Negatives

    return confusion_matrix
confusion_matrix = compute_confusion_matrix(TP, TN, FP, FN)

print(confusion_matrix)

#pickling
import pickle
model_filename = 'decision_tree_final.pkl'

with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {model_filename}")
