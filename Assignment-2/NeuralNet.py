#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

# data: https://archive.ics.uci.edu/dataset/53/iris
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.neural_network as nn
import sklearn.metrics
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)




    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization, categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input
        # print(type(self.processed_data))

        # drop null values
        self.processed_data.dropna(inplace=True)
        # remove duplicates
        self.processed_data.drop_duplicates(inplace=True)
        # categorical to numerical
        if "class" in self.processed_data:
            class_map = {}
            unique_val = self.processed_data["class"].unique()
            for index, class_label in enumerate(unique_val):
                class_map[class_label] = index
            self.processed_data["class"]=self.processed_data["class"].map(class_map)

        numerical_columns = ["sepal length", "sepal width", "petal length", "petal width"]
        # normalize by turning values to a specific range, 0-1
        for col in numerical_columns:
            # print(self.processed_data[col])
            min = self.processed_data[col].min()
            max = self.processed_data[col].max()
            self.processed_data[col] = (self.processed_data[col] - min) / (max - min)
        # standardization
        for col in numerical_columns:
            mean = self.processed_data[col].mean()
            std = self.processed_data[col].std()
            self.processed_data[col] = (self.processed_data[col] - mean) / std
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance metrics

        # Plot the model history for each model in a single plot model history is a plot of accuracy vs number of epochs you may want to create a large sized plot to show multiple lines in a same figure.

        results = []
        for activation in activations:
            for lr in learning_rate:
                for mi in max_iterations:
                    for nhl in num_hidden_layers:
                        model = nn.MLPClassifier(hidden_layer_sizes=(100,)*nhl, activation=activation, learning_rate_init=lr, max_iter=500)
                        model.fit(X_train, y_train)

                        # mse
                        train_predictions = model.predict(X_train)
                        test_predictions = model.predict(X_test)
                        train_mse =sklearn.metrics.mean_squared_error(y_train, train_predictions)
                        test_mse = sklearn.metrics.mean_squared_error(y_test, test_predictions)

                        # predictions
                        train_accuracy = sklearn.metrics.accuracy_score(y_train, model.predict(X_train))
                        test_accuracy = sklearn.metrics.accuracy_score(y_test, model.predict(X_test))

                        results.append({
                            "activation": activation,
                            "learning rate": lr,
                            "max iterations": mi,
                            "number of hidden layers": nhl,
                            "train accuracy": train_accuracy,
                            "test accuracy": test_accuracy,
                            "train MSE": train_mse,
                            "test MSE": test_mse
                        })
                        actual_epochs = len(model.loss_curve_)
                        # plotting the model to be shown
                        plt.plot(range(1, actual_epochs + 1), model.loss_curve_, label=f'{activation}, lr={lr}, epochs={actual_epochs}, layers={nhl}')

        plt.title('Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        results_data_frame = pd.DataFrame(results)
        print(results_data_frame)
        plt.show()
        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://raw.githubusercontent.com/abayindir1/ML-Assignments/master/Assignment-2/iris.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
