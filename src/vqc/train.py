
import numpy as py 
import matplotlib.pyplot as plt #draws and saves the training-loss chart 

#variables:
train = any 
np = any

import configparser #holds contants (settings file)
from data_loader import load_ad_hoc_dataset #imports the traning and tesst data 
from model import predict_expectation, predict_label, compute_loss #predict_expectation: runs the quantum circuit, predict_label: turns the value into 0 or 1, compute_loss: compares predictions to true labels and returns a loss number 
from optimizer import train #runs the classical optimization loop 
from quantum_circuits import geta_ansatz #builds ansatz (the circuit) and lists its parameters 

def make_loss_function(x_train, y_train): #function the optimizer will call 
   
    def loss (theta): 

        y_pred = [predict_expectation(x, theta) for x in x_train] #get a prediction for each training point 
        return compute_loss(y_train, y_pred, kind="mse") #compares predictions to real answers 

    return loss 

def evaluate_accuracy(x_data, y_true, theta) -> float: 

    y_pred = [predict_label(x,theta) for x in x_data] #predicts a label for each data point 
    return float(np.mean(np.array(y_pred) == np.array(y_true))) # % of predictions that match the real labels 

def main():

    np.random.seed(configparser.RANDOM_SEED) #same random numbers every run

    #Step 1 - makes the training and test data
    x_train, y_train, x_test, = load_ad_hoc_dataset(
        training_size=20, test_size=20, n=configparser.N_QUBITS, gap=0.3
    )

    #Step 2: initial parameters theta 
    theta_params = geta_ansatz(configparser.N_QUBITS, configparser.ANSATZ_REPS) #builds the circuit and gets its parameters 
    theta_init = np.random.uniform (0, 2 * np.pi, size=len(theta_params)) #starts with random values for those parameters

    #Steps 3-6: train
    loss_fn = make_loss_function(x_train, y_train) #set up the loss functionn using our data
    theta_star, loss_history = train(theta_init, loss_fn, method= "spsa", max_iters=configparser.MAX_ITERS) #trian the model and get the best parameters and loss over time 

    #Step 7: evaluate and plot 
    train_acc = evaluate_accuracy(x_train, y_train, theta_star) #checks the accuracy on the trianing data 
    test_acc = evaluate_accuracy(x_train, y_train, theta_star) #checks the accuracy on the test data
    print (f"Train accuracy: {train_acc:.2%}") #prints training accuracy score 
    print (f"Train accuracy: {test_acc:.2%}") #prints testing accuracy score  

    plt.plot(loss_history) #plots how the loss changed during training 
    plt.xlabel("Iteration")
    plt.ylabel ("Traning Loss")
    plt.savefig("tranin_curve.png")
    plt.show() 

if__name__ == "__main__": main() #only run main if this file is run directly 
