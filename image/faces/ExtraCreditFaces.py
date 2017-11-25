import os
import math


def get_training_data():
    digit_dir = os.getcwd()
    training_images = digit_dir + "/facedatatrain"
    new_text = open(training_images, "r")
    train_data = []
    current_digit_reading_in = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 70:
            line_counter += 1
            to_append = []
            for character in line:
                if character == ' ':
                    to_append.append('')
                elif character == '\n':
                    pass
                else:
                    to_append.append(character)
            current_digit_reading_in.append(to_append)
        if line_counter == 70:
            line_counter = 0
            train_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    training_labels = digit_dir + "/facedatatrainlabels"
    new_labels = open(training_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return train_data, labels

def probability_of_priors(labels):
    probabilities = 0
    for i in labels:
        if i ==1:
            probabilities+= 1
    probabilities = probabilities/ len(labels)
    return probabilities


def pixel_likelihoods(faces_input, laplace_constant=1):
    length = len(faces_input)
    likelihood = [0] * 70
    for i in range(0, 70):
        likelihood[i] = [laplace_constant] * 60
    for sample in faces_input:
        for i in range(0, 70):
            for j in range(0, 60):
                if sample[i][j] == '#':
                    likelihood[i][j] += 1

    for i in range(0, 70):
        for j in range(0, 60):
            likelihood[i][j] /= (length + laplace_constant * 2)

    for row in likelihood:
        new_row=[]
        formatted_row = ['%.3f' % elem for elem in row]
        new_row.append(formatted_row)
        print(new_row)
    return likelihood

def get_testing_data():
    digit_dir = os.getcwd()
    testing_images = digit_dir + "/facedatatest"
    new_text = open(testing_images, "r")
    test_data = []
    current_digit_reading_in = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 70:
            line_counter += 1
            to_append = []
            for character in line:
                if character == ' ':
                    to_append.append(' ')
                elif character == '\n':
                    pass
                else:
                    to_append.append(character)
            current_digit_reading_in.append(to_append)
        if line_counter == 70:
            line_counter = 0
            test_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    testing_labels = digit_dir + "/facedatatestlabels"
    new_labels = open(testing_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return test_data, labels


def map_calculation(test_digit, likelihood_list, prior):
    current_prob_pos = math.log(prior)
    current_prob_neg = math.log(prior)
    for i in range(0, 60):
        for j in range(0, 60):
            chance_of_edge_here = likelihood_list[i][j]
            if test_digit[i][j] == '#':
                current_prob_pos += math.log(chance_of_edge_here)
            else:
                current_prob_pos += math.log((1-chance_of_edge_here))

            if test_digit[i][j] != '#':
                current_prob_neg += math.log(1-chance_of_edge_here)
            else:
                current_prob_neg += math.log((chance_of_edge_here))
    if current_prob_pos>current_prob_neg:
        return(1)
    if current_prob_neg<=current_prob_neg:
        return(0)


def overall_accuracy(testingData, list_of_likelihood, prior, testingLabels):
    correct = 0

    confusion_matrix = [0] * 2
    for i in range(0, 2):
        confusion_matrix[i] = [0] * 2

    for i in range(0, len(testingData)):
        generated_value = map_calculation(testingData[i], list_of_likelihood, prior)
        actual_value = testingLabels[i]
        confusion_matrix[actual_value][generated_value] += 1
        if generated_value == actual_value:
            correct += 1
    correct /= len(testingLabels)
    return (confusion_matrix, correct)


if __name__ == "__main__":
    trainingData, trainingLabels = get_training_data()
    testingData, testingLabels = get_testing_data()
    prior = probability_of_priors(trainingLabels)
    print(prior)
    list_of_likelihood = pixel_likelihoods(trainingData)
    confusion_matrix, overall_probablility = overall_accuracy(testingData, list_of_likelihood, prior, testingLabels)
    print(overall_probablility)

    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.2f' % elem for elem in row]
        print(row)
