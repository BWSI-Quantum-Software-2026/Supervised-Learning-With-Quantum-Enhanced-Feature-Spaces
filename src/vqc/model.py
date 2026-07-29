import numpy as np
from qiskit.primitives import StatevectorSampler as Sampler
from src.feature_map import feature_map_circuit
from src.vqc.ansatz import ansatz
from src.vqc.optimizer import train # this is the classical optimizer loop that fit() uses
from qiskit.circuit import ParameterVector




class VQCModel:
    def __init__(self, qubits, repetitions): #starts of my class with the parameters and circuits that i need
        self.num_qubits = qubits
        self.reps = repetitions
   
        self.x = ParameterVector('x', qubits)
        self.feature_map = feature_map_circuit(np.array(list(self.x)))        
        self.ansatz, self.theta = ansatz(qubits, repetitions)
        
       
        self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit.measure_all()
    
        self.sampler = Sampler(seed = 42)  # make it so that its seeded so results are the same

    def calculate_parity(self, bitstring: str) -> int: #looks at characters 
        """
        looks through bitstring that i get from circuit and then checks if number is even or odd 
        even 1's: +1
        odd 1's: -1
        the significance is that it is impotrant in context of measuring z^n pauli observable
        """
        count_ones = bitstring.count('1')

        if count_ones%2 == 0:
            return 1
        else:
            return -1
    def expectation_value(self, x_val, theta_val):
        """
        uses x and theta ti find the expectation value in the range [-1,1]
        """
        param_dict={}
        for i in range(len(self.x)):
            param_symbol = self.x[i]
            num = x_val[i]
            param_dict[param_symbol] = num
        for i in range(len(self.theta)):
            param_symbol = self.theta[i]
            num = theta_val[i]
            param_dict[param_symbol] = num

        bound_circuit = self.circuit.assign_parameters(param_dict)

        job = self.sampler.run([bound_circuit], shots = 1024)#change the number of shots to what it is supposed to be
        result = job.result()[0]
        counts = result.data.meas.get_counts()

        total_shots = 1024
        expectation_val = 0.0

        for bitstring, count in counts.items():
            prob = count/total_shots
            expectation_val += self.calculate_parity(bitstring) * prob
        return expectation_val

    def label(self, x_val, theta_val) -> int:

        """
        convert between the expected value into traditional hard binary class
        """
        score = self.expectation_value(x_val, theta_val)

        if score >= 0.0:
            return 1
        else:
            return -1

    def fit(self, X, y, method="spsa", max_iters=50, seed=42):
        # needed to find the theta that gets the most labels right. sklearn does this step for kernel side and we have to run the optimizer ourselves

        rng = np.random.default_rng(seed)
        theta_start = rng.uniform(0, 2 * np.pi, len(self.theta)) # this just makes sure it starts at at random angles

        def loss(theta): # optimizer.py keeps calling this with different theta
            scores = [self.expectation_value(x, theta) for x in X]
            return float(np.mean((np.array(y) - np.array(scores)) ** 2)) # the squared error vs the real labels

        self.theta_star, self.history = train(theta_start, loss, method=method, max_iters=max_iters)
        return self

    def score(self, X, y) -> float:
        # fraction of labels we get right. needs fit() to have run first
    
        if not hasattr(self, "theta_star"):
            raise RuntimeError("call fit() before score()")

        predictions = [self.label(x, self.theta_star) for x in X]
        return float(np.mean(np.array(predictions) == np.array(y)))

        
