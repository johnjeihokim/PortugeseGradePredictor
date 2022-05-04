import numpy as np
import network
import csv



def toNums(x):
    """
    x is the row from the csv file
    Let it return the ndarray (32 X 1) with dtype float64
    """
    retval = np.ndarray(shape=(32,1), dtype = np.float64)

    for count, data in enumerate(x):
        if count == 31: break
        if (count == 0):
            retval[count][0] = 1.0 if data == "GP" else 0.0
        elif (count == 1):
            retval[count][0] = 1.0 if data == "M" else 0.0
        elif (count == 2): 
            retval[count][0] = np.float64(data) / 22.0
        elif (count == 3):
            retval[count][0] = 1.0 if data == "U" else 0.0
        elif (count == 4):
            retval[count][0] = 0.0 if data == "LE3" else 1.0
        elif (count == 5):
            retval[count][0] = 0.0 if data == "T" else 1.0
        elif count >= 6 and count <= 7:
            retval[count][0] = np.float64(data) / 4.0
        elif count == 8 or count == 9:
            if (data == "teacher"): retval[count][0] = 0.0
            elif (data == "health"): retval[count][0] = 1.0
            elif (data == "services"): retval[count][0] = 2.0
            elif (data == "at_home"): retval[count][0] = 3.0
            else: retval[count][0] = 4.0
            retval[count][0] = retval[count][0] / 4.0
        elif count == 10:
            if (data == "home"): retval[count][0] = 0.0
            elif (data == "reputation"): retval[count][0] = 1.0
            elif(data == "course"): retval[count][0] = 2.0
            retval[count][0] = retval[count][0]/2.0
        elif count == 11:
            if (data == "mother"): retval[count][0] = 0.0
            elif (data == "father"): retval[count][0] = 1.0
            elif (data == "other"): retval[count][0] = 2.0
            retval[count][0] = retval[count][0]/2.0
        elif count == 12 or count == 13 or count == 14:
            retval[count][0] = np.float64(data)
            retval[count][0] = retval[count][0]/4.0
        elif count >= 15 and count <= 22:
            if (data == "no"): retval[count][0] = 0.0
            else: retval[count][0] = 1.0
        elif count >= 23 and count <= 28:
            retval[count][0] = np.float64(data) / 5
        elif count == 29:
            retval[count][0] = np.float64(data)/ 93
        elif count == 30 or count == 31:
            retval[count][0] = np.float64(data) / 20
        
        return retval

def main():
    # get training data and test data
    training_data = []
    test_data = []

    with open('student-por.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';') 
        for outside_count,row in enumerate(spamreader):
            if outside_count == 0: continue
            
            input_vector = np.ndarray(shape = (32,1), dtype = np.float64)

            for count, data in enumerate(row):
                if count == 32: break
                if (count == 0):
                    input_vector[count][0] = 1.0 if data == "GP" else 0.0
                elif (count == 1):
                    input_vector[count][0] = 1.0 if data == "M" else 0.0
                elif (count == 2): 
                    input_vector[count][0] = np.float64(data) / 22.0
                elif (count == 3):
                    input_vector[count][0] = 1.0 if data == "U" else 0.0
                elif (count == 4):
                    input_vector[count][0] = 0.0 if data == "LE3" else 1.0
                elif (count == 5):
                    input_vector[count][0] = 0.0 if data == "T" else 1.0
                elif count >= 6 and count <= 7:
                    input_vector[count][0] = np.float64(data) / 4.0
                elif count == 8 or count == 9:
                    if (data == "teacher"): input_vector[count][0] = 0.0
                    elif (data == "health"): input_vector[count][0] = 1.0
                    elif (data == "services"): input_vector[count][0] = 2.0
                    elif (data == "at_home"): input_vector[count][0] = 3.0
                    else: input_vector[count][0] = 4.0
                    input_vector[count][0] = input_vector[count][0] / 4.0
                elif count == 10:
                    if (data == "home"): input_vector[count][0] = 0.0
                    elif (data == "reputation"): input_vector[count][0] = 1.0
                    elif(data == "course"): input_vector[count][0] = 2.0
                    input_vector[count][0] = input_vector[count][0]/2.0
                elif count == 11:
                    if (data == "mother"): input_vector[count][0] = 0.0
                    elif (data == "father"): input_vector[count][0] = 1.0
                    elif (data == "other"): input_vector[count][0] = 2.0
                    input_vector[count][0] = input_vector[count][0]/2.0
                elif count == 12 or count == 13 or count == 14:
                    input_vector[count][0] = np.float64(data)
                    input_vector[count][0] = input_vector[count][0]/4.0
                elif count >= 15 and count <= 22:
                    if (data == "no"): input_vector[count][0] = 0.0
                    else: input_vector[count][0] = 1.0
                elif count >= 23 and count <= 28:
                    input_vector[count][0] = np.float64(data) / 5
                elif count == 29:
                    input_vector[count][0] = np.float64(data)/ 93
                elif count == 30 or count == 31:
                    input_vector[count][0] = np.float64(data) / 20

            output_vector = np.zeros(shape = (21, 1), dtype = np.float64)
            the_one = int(row[-1])
            output_vector[the_one] = 1.0
            training_tuple = (input_vector, output_vector)
            training_data.append(training_tuple)

            if outside_count <= 400 and outside_count >= 201:
                test_tuple = (input_vector, the_one)
                test_data.append(test_tuple)

    # initialize network object net
    net = network.Network([32,16,21])

    #training the network
    net.SGD(training_data, 1000, 5, 1.0, test_data = test_data)


if __name__ == "__main__":
    main()