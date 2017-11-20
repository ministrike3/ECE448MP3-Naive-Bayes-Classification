from trainingFunctions import *

def get_testing_data():
    digit_dir = os.getcwd() + "/testData/"
    testing_images = digit_dir + "testimages"
    new_text = open(testing_images, "r")
    test_data = []
    current_digit_reading_in = []
    line_counter = 0
    for line in new_text.readlines():
        if line_counter != 28:
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
        if line_counter == 28:
            line_counter = 0
            test_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    testing_labels = digit_dir + "testlabels"
    new_labels = open(testing_labels, "r")
    labels = []
    for line in new_labels.readlines():
        labels.append(int(line))
    return test_data, labels

def map_calculation(test_digit,likelihood_list,prior):
    bayes_calcs=[]
    for x in range(0,10):
        current_prob=prior[x]
        for i in range (0,28):
            for j in range(0,28):
                chance_of_1_here=likelihood_list[x][i][j]
                if test_digit[i][j]==' ':
                    current_prob *= (1-chance_of_1_here)
                else:
                    current_prob *= chance_of_1_here
        bayes_calcs.append(current_prob)
        
    value=bayes_calcs.index(max(bayes_calcs))
    #print(value)
    return(value)

def overall_accuracy(testingData,list_of_likelihood,prior,testingLabels):
    correct=0
    for i in range (0,len(testingData)):
        generated_value=map_calculation(testingData[i], list_of_likelihood, prior)
        actual_value=testingLabels[i]
        if actual_value==generated_value:
            correct+=1
    correct=correct/len(testingData)
    return(correct)

if __name__ == "__main__":
    trainingData, trainingLabels = get_training_data()
    organ = organize_training_data(trainingData, trainingLabels)
    prior = probability_of_priors(trainingLabels)
    list_of_likelihood = pixel_likelihoods(organ)
    testingData, testingLabels = get_testing_data()
    print(overall_accuracy(testingData,list_of_likelihood,prior,testingLabels))