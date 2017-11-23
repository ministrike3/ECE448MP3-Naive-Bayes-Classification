import os
from disjoint_conversions import *
def get_training_data():
    digit_dir = os.getcwd() + "/trainingData/"
    training_images = digit_dir + "trainingimages"
    new_text = open(training_images, "r")
    train_data = []
    current_digit_reading_in = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 28:
            line_counter += 1
            to_append = []
            for character in line:
                if character == ' ':
                    to_append.append('0')
                elif character == '\n':
                    pass
                else:
                    to_append.append('1')
            current_digit_reading_in.append(to_append)
        if line_counter == 28:
            line_counter = 0
            train_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    training_labels = digit_dir + "traininglabels"
    new_labels = open(training_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return train_data, labels

def organize_training_data(training_data, training_labels):
    organized_digits = []
    for i in range(0, 10):
        organized_digits.append([])
    for i in range(0, 5000):
        data = training_data[i]
        label = training_labels[i]
        organized_digits[label].append(data)
    return organized_digits


def probability_of_priors(labels):
    probabilities = [0] * 10
    for i in labels:
        probabilities[i] += 1
    for i in range(0, 10):
        probabilities[i] = probabilities[i] / len(labels)
    return probabilities

if __name__ == "__main__":
    training_data, training_labels = get_training_data()
    disjoint_featuring_conversion_2_2(training_data,2,2)