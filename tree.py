import numpy as np
import csv

# https://github.com/random-forests used as reference.


# The headers for each column.
header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class_val"]

results = []


# Load the data from CSV files as arrays.
def get_data():
    ifile = open('car.training.csv', 'rU')
    reader = csv.reader(ifile, delimiter=',')

    rownum = 0
    training_data = []
    for row in reader:
        training_data.append(row)
        rownum += 1

    ifile.close()

    ifile = open('car.test.csv', 'rU')
    reader = csv.reader(ifile, delimiter=',')

    rownum = 0
    testing_data = []
    for row in reader:
        testing_data.append(row)
        rownum += 1

    ifile.close()

    return training_data, testing_data


# Calculate unique values in each row.
def unique_vals(rows, cols):
    return set([row[col] for row in rows])


# Calculate how many times each unique label has appeared in a row.
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# Check if a value is numeric
def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


# Class to create the best question after calculating info gain.
class Question:
    def __init__ (self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__  (self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition,
        str(self.value))


# Function to calculate rows after splitting to True and False.
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# Function to calculate GINI.
def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob = counts[label] / float(len(rows))
        impurity -= prob**2
    return impurity


# Function to calculate Information Gain.
def info_gain(left, right, current):
    p = float(len(left)) / (len(left) + len(right))
    return current - p * gini(left) - (1 - p) * gini(right)


# Function to calculate best possible split for decision tree.
def best_split(rows):
    best_gain = 0
    best_question = None
    current = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows)== 0:
                continue
            gain = info_gain(true_rows, false_rows, current)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    def __init__ (self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__ (self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# A recursive function to build the tree.
def build_tree(rows):
    gain, question = best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


# Print the tree.
def print_tree(node, spacing = ""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "    ")

    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


# Print the percentage accuracies of each prediction.
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100))
        results.append(probs[lbl])
    return probs


# Classify each prediction as acc or unacc.
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)



if __name__ == '__main__':
    training_data, testing_data = get_data()

    my_tree = build_tree(training_data)
    print_tree(my_tree)

    total = 0
    true = 0

    for row in testing_data:
        # print ("Actual: %s. Predicted: %s" %
        # (row[-1], print_leaf(classify(row, my_tree))))
        total += 1
        if (all (row[-1] == a for a in (list((print_leaf(classify(row, my_tree))))))):
            true += 1
    print("\n\nAccuracy :", true/total * 100)
