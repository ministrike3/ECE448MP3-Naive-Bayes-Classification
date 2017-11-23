import os
from part2_conversion_functions import *


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
    number_of_rows = len(organized_data[0][0])
    number_of_columns = len(organized_data[0][0][0])
    number_of_possible_values = 2 ** len(organized_data[0][0][0][0])
    list_of = []
    for digit in organized_data:
        length = len(digit)
        likelihood = [0] * number_of_rows
        for i in range(0, number_of_rows):
            likelihood[i] = [laplace_constant] * number_of_columns
            for x in range(0, number_of_columns):
                likelihood[i][x] = [laplace_constant] * number_of_possible_values
        for sample in digit:
            for i in range(0, number_of_rows):
                for j in range(0, number_of_columns):
                    current = int(sample[i][j], 2)
                    likelihood[i][j][current] += 1

        for i in range(0, number_of_rows):
            for j in range(0, number_of_columns):
                for k in range(0, number_of_possible_values):
                    likelihood[i][j][k] /= (length + laplace_constant * 10)
        list_of.append(likelihood)
    return list_of


def get_test_data():
    digit_dir = os.getcwd() + "/testData/"
    test_images = digit_dir + "testimages"
    new_text = open(test_images, "r")
    test_data = []
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
            test_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    test_labels = digit_dir + "testlabels"
    new_labels = open(test_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return test_data, labels


def map_calculation(test_digit, likelihood_list, prior):
    bayes_calcs = []
    for x in range(0, 10):
        current_prob = prior[x]
        for i in range(0, len(test_digit)):
            for j in range(0, len(test_digit[0])):
                what_is_here_in_test_digit = test_digit[i][j]
                chance_of_this = likelihood_list[x][i][j][int(what_is_here_in_test_digit, 2)]
                current_prob *= chance_of_this
        bayes_calcs.append(current_prob)

    value = bayes_calcs.index(max(bayes_calcs))
    # print(value)
    return (value)


def overall_accuracy(testingData, list_of_likelihood, prior, testingLabels):
    correct = 0
    confusion_matrix = [0] * 10
    for i in range(0, 10):
        confusion_matrix[i] = [0] * 10
    for i in range(0, len(testingData)):
        generated_value = map_calculation(testingData[i], list_of_likelihood, prior)
        actual_value = testingLabels[i]
        confusion_matrix[actual_value][generated_value] += 1
        if generated_value == actual_value:
            correct += 1
    correct /= len(testingLabels)
    return (confusion_matrix, correct)


if __name__ == "__main__":
    train_data, train_labels = get_training_data()
    overlapping_featuring_conversion_2_3(train_data)
    sorted_by_number = organize_training_data(train_data, train_labels)
    priors = probability_of_priors(train_labels)
    blah = pixel_likelihoods(sorted_by_number,0.1)
    test_data, test_labels = get_test_data()
    overlapping_featuring_conversion_2_3(test_data)
    confusion_matrix, overall_probablility = overall_accuracy(test_data, blah, priors, test_labels)
    print(overall_probablility)
    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.2f' % elem for elem in row]
        print(formatted_row)
