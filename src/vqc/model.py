
import numpy as np
from qiskit.primitives import StatevectorSampler as Sampler
from src.feature_map import apply
from src.vqc.ansatz import ansatz
self: VQCModel



class VQCModel:
    def initialize(self, qubits, repetitions): #starts of my class with the parameters and circuits that i need
        self.num_qubits = qubits
        self.reps = repetitions
   
        self.feature_map, self.x = apply(qubits, repetitions)
        self.ansatz, self.theta = ansatz(qubits, repetitions)
        
       
        self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit.measure_all()
        
        self.sampler = Sampler()

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
            exepctation_val += self.calculate_parity(bitstring) * prob
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

        
