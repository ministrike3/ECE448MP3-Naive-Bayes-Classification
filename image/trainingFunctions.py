import os

def get_training_data():
    digit_dir=os.getcwd()+"/trainingData/"
    training_images=digit_dir+"trainingimages"
    new_text=open(training_images, "r")
    train_data=[]
    current_digit_reading_in=[]
    line_counter=0
    for line in new_text.readlines():
        if line_counter!=28:
            line_counter+=1
            to_append=[]
            for character in line:
                if character==' ':
                    to_append.append(' ')
                elif character=='\n':
                    pass
                else:
                    to_append.append(character)
            current_digit_reading_in.append(to_append)
        if line_counter==28:
            line_counter = 0
            train_data.append(current_digit_reading_in)
            current_digit_reading_in = []
    training_labels = digit_dir + "traininglabels"
    new_labels=open(training_labels, "r")
    labels=[]
    for line in new_labels.readlines():
        labels.append(int(line))
    return train_data, labels

def organize_training_data(training_data,training_labels):
    organized_digits=[]
    for i in range(0,10):
        organized_digits.append([])
    for i in range(0,5000):
        data=training_data[i]
        label=training_labels[i]
        organized_digits[label].append(data)
    return(organized_digits)

def probability_of_priors(labels):
    probs=[0]*10
    for i in labels:
        probs[i]+=1
    for i in range(0,10):
        probs[i]=probs[i]/len(labels)
    return probs

def pixel_likelyhoods(organized_data):
    list_of_likelyhoods=[]
    for digit in organized_data:
        likelyhood = [0] * 28
        for i in range(0,28):
            likelyhood[i] = [0] * 28

        for sample in digit:
            for i in range(0,28):
                for j in range(0,28):
                    if sample[i][j]!=' ':
                        likelyhood[i][j]+=1

        for row in likelyhood:
            print(row)
        list_of_likelyhoods.append(likelyhood)
    return(list_of_likelyhoods)

if __name__ == "__main__":
    training_data,training_labels = get_training_data()
    organ=organize_training_data(training_data,training_labels)
    prior=probability_of_priors(training_labels)
    list_of_likelyhood=pixel_likelyhoods(organ)