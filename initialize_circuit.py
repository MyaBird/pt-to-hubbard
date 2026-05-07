import pennylane as qml
import numpy as np

class InitializeCircuit:

    '''Class to create blank circuit with dimensionality N + M + 2.

    N refers to the number of qubits required for a system of size 2^N
    M refers to the number of ancillary qubits required to implement the U_e gate
    The 2 additional qubits are required for readout

    '''

    def __init__(self, N, M):
        '''Constructor

        Parameters:
        --------------------------
        N :int
            Describes number of qubits representative a system of size 2^N

        M :int
            Number of ancillary qubits required to implement the U_e gate

        This function creates two additional qubits for readout purposes, required to test the success of the perturbation gates and U_e gates.
        These qubits are denoted q1'' and q2''
        '''

        self.N_qubits = []
        self.N = N
        self.M_qubits = []
        self.M = M
        self.readout_qubits = ["q1''", "q2''"]

        #Define qubits representative of system as qn
        for num in range(N):
            self.N_qubits.append('q' + str(num+1))

        #Define ancilla qubits for constructing U_e as qm'
        for num in range(M):
            self.M_qubits.append('q' + str(num+1)+"'")

        #Combine system qubits, ancillas, and readout qubits to define all qubits used for algorithm
        self.qubits = self.N_qubits + self.M_qubits + self.readout_qubits

    def create_device(self, device:str):
        '''Create the computational device used for quantum circuit'''

        return qml.device(device, wires=self.qubits)

    def initial_state(self, config:list):
        '''Set the initial state of system (first N qubits)
        
        requirements:
        - list should be of length N
        - list should contain objects of only integer values 1 or 0, corresponding to |1> or |0> state
        
        '''

        #Check that config input matches dimensionality of system qubits
        if len(config) != self.N:
            error("initial configuration has wrong dimension")

        #Apply Pauli X gate to wire depending if corresponding config index is 1
        for ind, bit in enumerate(config):
            if bit == 1:
                qml.PauliX(wires=self.qubits[ind])
            else:
                pass