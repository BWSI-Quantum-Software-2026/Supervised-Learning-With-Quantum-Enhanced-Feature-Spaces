import numpy as np 
import matplotlib.pyplot as plt #draws and saves the training-loss chart 

from src.datasets import load_adhoc #imports the traning and tesst data 
from src.model import predict_expectation, predict_label #predict_expectation: runs the quantum circuit, predict_label: turns the value into 0 or 1 
from src.optimizer import train #runs the classical optimization loop 
from src.ansatz import ansatz #builds ansatz (the circuit) and lists its parameters 

def compute_loss(y_true, y_pred): #compares predictions to true labels and returns a loss number 
    #mean squared error. y_pred are the raw scores in [-1,1] and y_true is -1/+1, so the
    #closer the score sits to the real label the smaller this gets
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def make_loss_function(x_train, y_train): #function the optimizer will call 
   
    def loss (theta): 

        y_pred = [predict_expectation(x, theta) for x in x_train] #get a prediction for each training point 
        return compute_loss(y_train, y_pred) #compares predictions to real answers 

    return loss 

def evaluate_accuracy(x_data, y_true, theta) -> float: 

    y_pred = [predict_label(x,theta) for x in x_data] #predicts a label for each data point 
    return float(np.mean(np.array(y_pred) == np.array(y_true))) # % of predictions that match the real labels 

def main():

    np.random.seed(42) #same random numbers every run

    #makes the training and test data
    x_train, y_train, x_test, y_test = load_adhoc(
        training_size=20, test_size=20, n=2, gap=0.3
    )

    #initial parameters theta 
    circuit, theta_params = ansatz(2, 2) #builds the circuit and gets its parameters 
    theta_init = np.random.uniform (0, 2 * np.pi, size=len(theta_params)) #starts with random values for those parameters

    #=train
    loss_fn = make_loss_function(x_train, y_train) #set up the loss functionn using our data
    theta_star, loss_history = train(theta_init, loss_fn, method= "spsa", max_iters=100) #trian the model and get the best parameters and loss over time 

    #=evaluate and plot 
    train_acc = evaluate_accuracy(x_train, y_train, theta_star) #checks the accuracy on the trianing data 
    test_acc = evaluate_accuracy(x_test, y_test, theta_star) #checks the accuracy on the test data
    print (f"Train accuracy: {train_acc:.2%}") #prints training accuracy score 
    print (f"Test accuracy: {test_acc:.2%}") #prints testing accuracy score  

    plt.plot(loss_history) #plots how the loss changed during training 
    plt.xlabel("Iteration") #x-axis
    plt.ylabel ("Traning Loss") #y-axis
    plt.savefig("traning_curve.png") #save chart as a image
    plt.show()