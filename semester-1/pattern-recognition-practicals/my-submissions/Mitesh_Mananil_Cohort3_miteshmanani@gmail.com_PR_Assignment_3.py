import pandas as pd
import numpy as np
import math

# Load the dataset
data = pd.read_csv(
    'C:\\Users\\mites\\OneDrive\\Desktop\\Mitesh Manani - archive\\Iris.csv')

# Convert relevant columns to numeric (ignoring the first column which is an ID)
for column in data.columns[1:5]:
    data[column] = pd.to_numeric(data[column])

# Separate the dataset by class


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset.iloc[i]
        class_value = vector.iloc[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated

# Summarize the dataset


def summarize_dataset(dataset):
    summaries = [(np.mean(column), np.std(column)) for column in zip(*dataset)]
    del summaries[-1]
    return summaries

# Summarize the data by class


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize_dataset(
            [instance[1:5] for instance in instances])
    return summaries

# Gaussian Probability Density Function


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

# Calculate class probabilities


def calculate_class_probabilities(summaries, input_vector):
    total_rows = sum([len(summaries[class_value])
                     for class_value in summaries])
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = len(
            summaries[class_value]) / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities

# Make a prediction


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Summarize the dataset by class
summaries = summarize_by_class(data)

# (a) Separate By Class
separated = separate_by_class(data)
print("Separated by class:")
for class_value in separated:
    print(f"{class_value}: {len(separated[class_value])} instances")

# (b) Summarize Dataset
summaries = summarize_dataset(
    data.iloc[:, 1:5].values)  # Use only numeric columns
print("\nDataset summary:")
print(summaries)

# (c) Summarize Data By Class
summaries_by_class = summarize_by_class(data)
print("\nData summary by class:")
for class_value, class_summaries in summaries_by_class.items():
    print(f"{class_value}: {class_summaries}")

# (d) Gaussian Probability Density Function
# Example calculation using the first attribute (sepal length) of the first instance in the dataset
# Use the mean and std deviation from the dataset summary for demonstration
mean, stdev = summaries[0]
x = data.iloc[0, 1]  # The sepal length of the first instance
probability = calculate_probability(x, mean, stdev)
print("\nGaussian Probability Density Function example:")
print(
    f"Value: {x}, Mean: {mean}, Standard Deviation: {stdev}, Probability: {probability}")

# (e) Class Probabilities
# Using a sample data point
input_vector = [5.7, 2.9, 4.2, 1.3]
probabilities = calculate_class_probabilities(summaries_by_class, input_vector)
print("\nClass probabilities for the input vector:")
print(probabilities)

# Predict the class
class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
predicted_class = predict(summaries_by_class, input_vector)
predicted_class_index = class_mapping[predicted_class]
print(f"\nData={input_vector}, Predicted: {predicted_class_index}")
