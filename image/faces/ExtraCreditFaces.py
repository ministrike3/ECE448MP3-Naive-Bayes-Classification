import os
from math import log


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
    probabilities = [0] * 2
    for i in labels:
        probabilities[i] += 1
    for i in range(0, 2):
        probabilities[i] = probabilities[i] / len(labels)
    return probabilities

def organize_training_data(training_data, training_labels):
    face_only = [[],[]]

    for i in range(0, len(training_labels)):
        label = training_labels[i]
        if label==1:
            face_only[1].append(training_data[i])
        if label==0:
            face_only[0].append(training_data[i])
    return face_only

def pixel_likelihoods(organized_data, laplace_constant=1):
    list_of = []
    for digit in organized_data:
        length = len(digit)
        likelihood = [0] * 70
        for i in range(0, 70):
            likelihood[i] = [laplace_constant] * 60
        for sample in digit:
            for i in range(0, 70):
                for j in range(0, 60):
                    if sample[i][j] == '#':
                        likelihood[i][j] += 1

        for i in range(0, 70):
            for j in range(0, 60):
                likelihood[i][j] /= (length + laplace_constant * 2)

        list_of.append(likelihood)

    return list_of

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


def map_calculation(test_digit, likelihood_list, prior, actual):
    bayes_calcs = []
    for x in range(0, 2):
        current_prob = log(prior[x])
        for i in range(0, 70):
            for j in range(0, 60):
                chance_of_1_here = likelihood_list[x][i][j]
                if test_digit[i][j] == '#':
                    current_prob += log(chance_of_1_here)
                else:
                    current_prob += log(1-chance_of_1_here)
        bayes_calcs.append(current_prob)

    value = bayes_calcs.index(max(bayes_calcs))
    # Added this variable calc and returned calc and value as a tuple (sid function)
    calc = bayes_calcs[actual]
    return (value, calc)


def overall_accuracy(testingData, list_of_likelihood, prior, testingLabels):
    correct = 0

    min_max_match = [0] * 2
    for i in range(0, 2):
        min_max_match[i] = [[0, 1], [0, -100000]]

    confusion_matrix = [0] * 2
    for i in range(0, 2):
        confusion_matrix[i] = [0] * 2

    for i in range(0, len(testingData)):

        actual_value = testingLabels[i]
        generated_value, calculated_chance = map_calculation(testingData[i], list_of_likelihood, prior, actual_value)
        if calculated_chance < min_max_match[actual_value][0][1]:
            min_max_match[actual_value][0][0] = i
            min_max_match[actual_value][0][1] = calculated_chance

        if calculated_chance > min_max_match[actual_value][1][1]:
            min_max_match[actual_value][1][0] = i
            min_max_match[actual_value][1][1] = calculated_chance

        confusion_matrix[actual_value][generated_value] += 1
        if generated_value == actual_value:
            correct += 1
    correct /= len(testingLabels)
    return (confusion_matrix, correct, min_max_match)


if __name__ == "__main__":
    trainingData, trainingLabels = get_training_data()
    test_data, test_labels = get_testing_data()
    prior = probability_of_priors(trainingLabels)
    print(prior)

    facelist=organize_training_data(trainingData, trainingLabels)

    list_of_likelihood = pixel_likelihoods(facelist,0.1)

    confusion_matrix, overall_probablility, minmax = overall_accuracy(test_data, list_of_likelihood, prior, test_labels)
    for i in range(0,len(minmax)):
        vals=minmax[i]
        min_index=vals[0][0]
        max_index=vals[1][0]
        print('index of least atypical', i, 'is', min_index*70,(min_index+1)*70)
        print('index of most atypical', i, 'is', max_index*70,(max_index+1)*70)
    print(overall_probablility)
    for x in range(0, len(confusion_matrix)):
        row = confusion_matrix[x]
        number_of_this_digit = sum(row)
        for i in range(0, len(row)):
            row[i] = row[i] / number_of_this_digit
        formatted_row = ['%.4f' % elem for elem in row]
        print(formatted_row)

