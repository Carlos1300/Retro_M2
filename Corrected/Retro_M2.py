"""

Author: Carlos de Jesús Ávila González
Title: Linear Regression Implementation
Date: 05/09/2022

"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def minMax(x):
    
        """
        This function normalize the data by applying the min-max normalization method.

        Args:
                x (ndarray) an array of the samples used for the model.

        Returns:
                scaled a list of the scaled samples after applying the min-max normalization method.
        """

        scaled = (x - np.min(x))/(np.max(x) - np.min(x))
        return scaled
    
def costFunction(thetas, x, y, m):

        """
        Calculates the cost function of the model.

        Args:
                thetas (lst) a list of the parameters we are using for our model.
                x (lst) a list of the samples used for the model.
                y (lst) a list of the true values of our model.

        Returns:
                J a number that represents the error of the model.
        """

        J = (1/(2*m)) * np.sum(((x @ thetas) - y) ** 2)

        return J

def gradientDescent(thetas, x, y, alpha, epochs):
        
        """
        Calculates the best parameters of the model (thetas) and the error of each
        new pair of thetas using the gradient descent method.
        
        Args:
                thetas (ndarray) an array of the initial parameters we are using for our model.
                x (ndarray) an array of the training samples of our model.
                y (ndarray) an array with the true values of our model.
                alpha (float) a float that indicates the learning rate or the movement towards
                                the local minimum of the model.
                epochs (int) an integer that specifies the number of epochs or iterations
                                where we will be perfoming the gradient descent method.
        
        Returns:
                thetas an array containing the best parameters for the model.
                errors an array containing the errors of the model.
        """
        
        errors = np.zeros([epochs, 1])

        for i in range(0,epochs):
                
                error = (x @ thetas) - y
                gradient = x.T @ error
                thetas -= (alpha/np.size(y)) * gradient
                
                errors[i] = costFunction(thetas, x, y, np.size(y))

        return thetas, errors

def results(X, x, y, new_thetas, errors):
        
        """
        This function prints the scatter plot and linear plot representing the result
        of applying the linear regression to our data. It also prints the plot that
        represent the error being minimized by the gradient descent method.

        Args:
                X (ndarray) an array containing the values of x before applying the
                                min-max method.
                x (ndarray) an array of the training samples of our model.
                y (ndarray) an array with the true values of our model.
                new_thetas (ndarray) an array of the parameters found by the gradient descent method.
                errors (ndarray) an array containing the errors of the gradient descent method.
                
        Returns:
                two plots that helps us visualize the results of the gradient descent method.
        """
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(X, y, s=10)
        plt.plot(X, x @ new_thetas, c="red")
        plt.title("Linear Regression")
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.subplot(1, 2, 2)
        plt.plot(errors, c="orange")
        plt.title("Gradient Descent")
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.show()
        
def predict(x, new_thetas):
        """
        This function predicts the value of x after the model has been trained.

        Args:
                x (ndarray) an array of the values we are goint to predict with our trained model.
                new_thetas (ndarray) an array of the parameters found by the gradient descent method.
                
        Returns:
                A predicted value.
        """
        
        return x @ new_thetas

def metrics(real, predicted):
        """
        This function calculates the R2-score for the model.

        Args:
                real (ndarray) an array of the real y values.
                predicted (ndarray) an array of the predicted values using the trained model.
                
        Returns:
                R2-score of the model
        """
        
        squared_error = np.sum((real - predicted) ** 2)
        total_errors = np.sum((real-np.mean(real))**2)
        
        return 1 - (squared_error/total_errors)

def modelInfo(new_thetas, performance):
        """
        This function prints the information of the trained model as well as the R2 score of it.

        Args:
                new_thetas (ndarray) an array containing the coefficient and the intercept of the model.
                performance (float) a float which indicates the R2 score of the model.
        """
        print("The intercept is: ", round(new_thetas[0][0], 2))
        print("The coefficient is: ", round(new_thetas[1][0], 2))
        print("The R2 score of the model is: ", round(performance*100, 2), "%")



        
################## PREPARATION OF THE MODEL #######################
        
thetas = np.zeros([2,1])  #Parameters of our model, in this case two because we are working with an univariate problem.
X_train = np.random.rand(100, 1) #Generate the independent variables
y_train = 2 + 3 * X_train + np.random.rand(100, 1)     #Generate the dependent variable.

X_train = minMax(X_train)   #Normalization of the x variable using the min-max method.
X_train = X_train.reshape([np.size(y_train), 1])   #Reshaping the array so we can work and input the x0 which value is 1
x_train = np.hstack([np.ones_like(X_train), X_train])     #Inputing the value of x0 in our x array

alpha = 0.01    #Learning rate for the model.
epochs = 500    #Epochs to run on the gradient descent algorithm.

################## GRADIENT DESCENT METHOD #######################

new_thetas, errors = gradientDescent(thetas, x_train, y_train, alpha, epochs) #Execute the gradient descent method.

################## PLOTTING THE MODEL AND THE ERRORS #######################

results(X_train, x_train, y_train, new_thetas, errors) #Plotting the results of the method.

################## PREDICTING WITH THE MODEL #######################

X_test = np.random.rand(30, 1)  #Creating the test data for our model (Representing the 30% of the train data).

y_test = 2 + 3 * X_test + np.random.rand(30, 1)        #Creating the test target for our model (Representing the 30% of the train data).

X_test = minMax(X_test)   #Normalization of the x variable using the min-max method.
X_test = X_test.reshape([np.size(y_test), 1])   #Reshaping the array so we can work and input the x0 which value is 1
x_test = np.hstack([np.ones_like(X_test), X_test])     #Inputing the value of x0 in our x array


predicted_values = predict(x_test, new_thetas) #Predicting new values

print(predicted_values) #Printing the predicted values

performance = metrics(y_test, predicted_values) #Getting the R2 score of the model

modelInfo(new_thetas, performance) #Printing the information and the R2 score of the model

