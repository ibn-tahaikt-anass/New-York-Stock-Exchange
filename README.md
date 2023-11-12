Neural Network Regression with PyTorch
This code implements a simple neural network regression model using PyTorch to predict stock prices. The dataset used for training and testing is loaded from a CSV file containing historical stock prices. The neural network architecture consists of three layers: the input layer, two hidden layers, and the output layer.


Dataset Preparation (CSVDataset class):

Loads a CSV file as a Pandas DataFrame.
Extracts input features (X) from columns 3 to 6 and target values (y) from the last column.
Splits the dataset into training and testing sets.
Neural Network Architecture (MLP class):

Implements a simple feedforward neural network with three layers.
The input layer has the same number of neurons as input features (4 in this case).
Two hidden layers with 10 and 8 neurons, respectively, and a ReLU activation function.
The output layer with one neuron for regression.
Data Preparation and Loading (prepare_data function):

Loads the dataset using the CSVDataset class.
Splits the dataset into training and testing sets.
Creates PyTorch data loaders for both sets.
Model Training (train_model function):

Uses Mean Squared Error (MSE) loss for regression.
Utilizes Stochastic Gradient Descent (SGD) as the optimization algorithm.
Iterates through epochs and mini-batches to train the neural network.
Model Evaluation (evaluate_model function):

Evaluates the trained model on the testing set.
Computes Mean Squared Error (MSE) as the evaluation metric.
Prediction (predict function):

Makes predictions on new data (a single row).

