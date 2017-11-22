import matplotlib.pyplot as plt
import numpy as np
from trainingFunctions import *
from testing import *
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

if __name__ == "__main__":
    trainingData, trainingLabels = get_training_data()
    testingData, testingLabels = get_testing_data()

    organ = organize_training_data(trainingData, trainingLabels)
    prior = probability_of_priors(trainingLabels)
    list_of_likelihood = pixel_likelihoods(organ, 0.1)
    plot_likelyhood(list_of_likelihood[3])
    plot_likelyhood(list_of_likelihood[4])
    plot_likelyhood(list_of_likelihood[5])
    plot_likelyhood(list_of_likelihood[7])
    plot_likelyhood(list_of_likelihood[8])
    plot_likelyhood(list_of_likelihood[9])

