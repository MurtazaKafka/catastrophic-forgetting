def compute_fisher_information(parameters, dataset, forward_function, loss_function):
    """
    Compute the Fisher Information Matrix (FIM) for a model.

    Args:
        parameters (list): A list of model parameters (assumed as floats or lists of floats).
        dataset (list): A list of tuples (inputs, targets) where inputs are model inputs and targets are the true outputs.
        forward_function (callable): A function representing the forward pass of the model. 
                                     Should take inputs and parameters as arguments and return predictions.
        loss_function (callable): A function to compute the loss. Takes predictions and targets as arguments.

    Returns:
        fisher_information (list): A list representing the Fisher Information Matrix for the parameters.
                                   Each entry corresponds to the Fisher value for the respective parameter.
    """
    # Initialize Fisher Information matrix with zeros
    fisher_information = [0.0 for _ in parameters]

    # Iterate over the dataset to compute gradients and contributions to Fisher Information
    for inputs, targets in dataset:
        # Forward pass: calculate predictions
        predictions = forward_function(inputs, parameters)

        # Compute loss
        loss = loss_function(predictions, targets)

        # Compute gradients of the loss with respect to each parameter
        gradients = []
        for i in range(len(parameters)):
            # Numerical derivative with respect to parameter i
            original_param = parameters[i]
            epsilon = 1e-6  # Small change for numerical differentiation
            
            # Compute loss with slightly increased parameter
            parameters[i] = original_param + epsilon
            loss_plus = loss_function(forward_function(inputs, parameters), targets)
            
            # Compute loss with slightly decreased parameter
            parameters[i] = original_param - epsilon
            loss_minus = loss_function(forward_function(inputs, parameters), targets)
            
            # Restore original parameter and compute gradient
            parameters[i] = original_param
            gradient = (loss_plus - loss_minus) / (2 * epsilon)
            gradients.append(gradient)

        # Accumulate Fisher Information contributions
        for i in range(len(parameters)):
            fisher_information[i] += gradients[i] ** 2

    # Normalize by the number of data points
    num_samples = len(dataset)
    fisher_information = [fi / num_samples for fi in fisher_information]

    return fisher_information

# Define a simple linear model as the forward function
def linear_model(inputs, parameters):
    # Return predictions as a list for consistency
    return [sum(x * p for x, p in zip(inputs, parameters))]

# Define a mean squared error loss function
def mean_squared_error(predictions, targets):
    # Ensure both predictions and targets are lists
    return sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(targets)

# Define dataset as a list of (inputs, targets) pairs
dataset = [
    ([1, 2, 3], [4]),
    ([2, 3, 4], [5]),
    ([3, 4, 5], [6]),
]

# Define initial model parameters
parameters = [0.5, 0.5, 0.5]

# Compute Fisher Information Matrix
fisher_matrix = compute_fisher_information(parameters, dataset, linear_model, mean_squared_error)

# Print results
print("Fisher Information Matrix:")
for i, fi in enumerate(fisher_matrix):
    print(f"Parameter {i}: {fi}")
