import matplotlib.pyplot as plt
import numpy as np
from part1 import *
from math import log

def plot_likelyhood(likelyhood_of_a_number):
    #here's our data to plot, all normal Python lists
    x = [i for i in range(0,len(likelyhood_of_a_number))]
    y = [i for i in range(0, len(likelyhood_of_a_number[0]))]

    intensity = likelyhood_of_a_number
    intensity.reverse()
    #setup the 2D grid with Numpy
    x, y = np.meshgrid(x, y)

    #convert intensity (list of lists) to a numpy array for plotting

    intensity = np.array(intensity)

    #now just plug the data into pcolormesh, it's that easy!
    xt=plt.pcolormesh(x, y, intensity)
    plt.colorbar() #need a colorbar to show the intensity scale
    xt.axes.get_xaxis().set_visible(False)
    xt.axes.get_yaxis().set_visible(False)
    plt.show() #boom

def odds_matrix_calculator(list_of_likelihoods,first,second):
    odds = [0] * 28
    for i in range(0, 28):
        odds[i] = [0] * 28
    first_likelihood=list_of_likelihoods[first]
    second_likelihood=list_of_likelihoods[second]
    for i in range(0,28):
        for j in range(0,28):
            odds[i][j]=log(first_likelihood[i][j]/second_likelihood[i][j])
    return(odds)

def pixel_likelihoods_log(organized_data, laplace_constant=1):
    list_of = []
    for digit in organized_data:
        length = len(digit)
        likelihood = [0] * 28
        for i in range(0, 28):
            likelihood[i] = [laplace_constant] * 28
        for sample in digit:
            for i in range(0, 28):
                for j in range(0, 28):
                    if sample[i][j] != ' ':
                        likelihood[i][j] += 1

        for i in range(0, 28):
            for j in range(0, 28):
                likelihood[i][j] /= (length + laplace_constant * 10)
                likelihood[i][j] = log(likelihood[i][j])

        list_of.append(likelihood)

        # for row in likelihood:
        #    print(row)
    return list_of



if __name__ == "__main__":
    trainingData, trainingLabels = get_training_data()
    testingData, testingLabels = get_testing_data()

    organ = organize_training_data(trainingData, trainingLabels)
    prior = probability_of_priors(trainingLabels)
    list_of_likelihood = pixel_likelihoods(organ, 0.1)
    # plot_likelyhood(list_of_likelihood[3])
    # plot_likelyhood(list_of_likelihood[4])
    # plot_likelyhood(list_of_likelihood[5])
    # plot_likelyhood(list_of_likelihood[7])
    # plot_likelyhood(list_of_likelihood[8])
    # plot_likelyhood(list_of_likelihood[9])
    plot_likelyhood(odds_matrix_calculator(list_of_likelihood,4,9))
    plot_likelyhood(odds_matrix_calculator(list_of_likelihood, 8, 3))
    plot_likelyhood(odds_matrix_calculator(list_of_likelihood, 5, 3))
    plot_likelyhood(odds_matrix_calculator(list_of_likelihood, 7, 9))


