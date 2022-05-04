import random
import csv

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [1]
        self.weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        dprod = np.dot(self.weights, a)
        retval = [sigmoid(dprod + self.biases[0])]
        return retval

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data) # 649
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.max(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x >= y -0.03 and x <= y + 0.03) for (x, y) in test_results)



    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def main():
    net = Network([31,1])

    trainingData = [] # list of tuples
    # trainingData can be represented as a list of (x,y)

    with open('student-por.csv') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';') 
        for count,row in enumerate(spamreader):
            if count == 0: continue
            expected = sigmoid(float(row[-1]))
            vector = np.ndarray(shape=(31,1), dtype=float)

            vector = toNums(row)

            trainingData.append((vector, expected))
    
    net.SGD(trainingData, 30, 10, 3.0, test_data = trainingData[100:200])


    # 649 is divisible by 11 and 59

def toNums(x):
    """
    x is the row from the csv file
    Let it return the ndarray (11 X 1)
    """
    retval = np.ndarray(shape=(31,1), dtype = float)
    for count, data in enumerate(x):
        if count == 31: break
        if (count == 0):
            retval[count] = 1.0 if data == "GP" else 0.0
        elif (count == 1):
            retval[count] = 1.0 if data == "M" else 0.0
        elif (count == 2): 
            retval[count] = float(data)
        elif (count == 3):
            retval[count] = 1.0 if data == "U" else 0.0
        elif (count == 4):
            retval[count] = 0.0 if data == "LE3" else 1.0
        elif (count == 5):
            retval[count] = 0.0 if data == "T" else 1.0
        elif count >= 6 and count <= 7:
            retval[count] = float(data)
        elif count == 8 or count == 9:
            if (data == "teacher"): retval[count] = 0.0
            elif (data == "health"): retval[count] = 1.0
            elif (data == "services"): retval[count] = 2.0
            elif (data == "at_home"): retval[count] = 3.0
            else: retval[count] = 4.0
        elif count == 10:
            if (data == "home"): retval[count] = 0.0
            elif (data == "reputation"): retval[count] = 1.0
            elif(data == "course"): retval[count] = 2.0
        elif count == 11:
            if (data == "mother"): retval[count] = 0.0
            elif (data == "father"): retval[count] = 1.0
            elif (data == "other"): retval[count] = 2.0
        elif count == 12 or count == 13 or count == 14:
            retval[count] = float(data)
        elif count >= 15 and count <= 22:
            if (data == "no"): retval[count] = 0.0
            else: retval[count] = 1.0
        elif count >= 23 and count <= 30:
            retval[count] = float(data)
        
        return retval
        

if __name__ == "__main__":
    main()
