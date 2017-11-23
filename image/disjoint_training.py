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


def pixel_likelihoods(organized_data, laplace_constant=1):
    list_of = []
    for digit in organized_data:
        length = len(digit)
        likelihood = [0] * 28
        for i in range(0, 28):
            likelihood[i] = [laplace_constant] * 28
            for x in range(0,28):
                likelihood[i][x]=[laplace_constant] * 3
        for sample in digit:
            for i in range(0, 28):
                for j in range(0, 28):
                    if sample[i][j] == ' ':
                        likelihood[i][j][0] += 1
                    if sample[i][j] == '+':
                        likelihood[i][j][1] += 1
                    if sample[i][j] == '#':
                        likelihood[i][j][2] += 1

        for i in range(0, 28):
            for j in range(0, 28):
                for k in range(0,3):
                    likelihood[i][j][k] /= (length + laplace_constant * 10)

        list_of.append(likelihood)

        # for row in likelihood:
        #     new_row=[]
        #     for item in row:
        #         formatted_row = ['%.2f' % elem for elem in item]
        #         new_row.append(formatted_row)
        #     print(new_row)
    return list_of


if __name__ == "__main__":
    training_data, training_labels = get_training_data()
    disjoint_featuring_conversion_2_4(training_data)

    for row in training_data[-1]:
        print(row)

    sorted_by_number=organize_training_data(training_data, training_labels)
    priors=probability_of_priors(training_labels)


    #disjoint_featuring_conversion_2_4(training_data)