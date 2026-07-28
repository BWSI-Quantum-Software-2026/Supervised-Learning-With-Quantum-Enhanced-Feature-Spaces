
#SPSA: optimization method - only needs 2 curcuit evaluations per step (regaurdless of how many parameters theta has)

#Functions (adjust based on the data set)
delta:any
theta:any 
loss_fn:any
c:any
a:any
gradient_estimate:any

import numpy as np 
import configparser #settings file (stores information)

def spasa_step (theta, loss_fn, a:float = configparser.LEARNING_RATE, c:float = configparser.SPSA_PERTURABATION): #SPSA - it wil nudge every parameter randomly, check if it helped, then step in whichever direction looked better

delta = np.random.choice([-1,1], size = len(theta)) #a coin flip for every parameter: +1 means try nudging it up and -1 meeans try nudging it down

loss_plus = loss_fn(theta + c * delta) #move in the direction "delta" points and check the error
loss_minus = loss_fn(theta - c * delta) #move in the OPPOSITE direction "delta" points and check the error

gradient_estimate = (loss_plus - loss_minus) / (2 * c) * delta #compare the two errors to guess whhoch way is downhill for each parameter at once
return theta - a * gradient_estimate #take a step down hill (learning rate "a")


