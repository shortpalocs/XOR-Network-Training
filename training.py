import numpy as np




X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1],
    ])


# Each of the 0's and 1's are Nodes


Y = np.array([
    [0],
    [1],
    [1],
    [0]
])




confidence_history = [[], [], [], []]
# Track confidence for all 4 of the X input values
# One list for each input value




lr = 0.1 # learning rate



# Random weight for hidden layer (2 neurons)
w1 = np.random.rand(2,2)
b1 = np.random.rand(1,2)


# Random weight for output layer (only 1 neuron)
w2 = np.random.rand(2,1) # multiplier, affects output based on it's value
b2 = np.random.rand(1,1) # offset




def train_step():
    global w1, w2, b1, b2
    # Forward pass stuff
    # Hidden layer
     # Multiply the input X by the weights w1 (X * w1), then add bias (b1) to it, final equation is (X * w1) + b1
    hidden_input = np.dot(X, w1) + b1  # np.dot multiplies 2 arrays together, X and w1 both being arrays
    hidden_output = 1 / (1 + np.exp(-hidden_input))  # Sigmoid activation

    output_input = np.dot(hidden_output, w2) + b2
    output = 1 / (1 + np.exp(-output_input))

    for i in range(4):
        confidence_history[i].append(output[i, 0])

    # Compute gradients (change each iteration)

    error = output - Y  # Difference between guess and true
    d_output = error * output * (1 - output)  # Derivative of sigmoid

    # Hidden layer error
    d_hidden = d_output.dot(w2.T) * hidden_output * (1 - hidden_output)

    # Update weights

    w2 -= lr * hidden_output.T.dot(d_output)
    b2 -= lr * np.sum(d_output, axis=0, keepdims=True)

    w1 -= lr * X.T.dot(d_hidden)
    b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True)


    current_step = len(confidence_history[0])

    # Print losses every 1k iterations

    if current_step % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"On step {current_step}, Loss: {loss}, Confidence at [0,0]: {output[0, 0]}")

    return current_step


