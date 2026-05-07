import pennylane as qml
import numpy as np
import math

from initialize_circuit import InitializeCircuit as cir

class ConstructionGates:

    '''Class to define all gates necessary for constructing circuit to estimate 1st order corrections to Hubbard Model
    '''

    def __init__(self, N, M, pert_coeff, timestep, u_param):

        '''Constructor

        Parameters:
        -----------------------------
        N :int
            Describes number of qubits representative of system

        M :int
            Described number of ancilla qubits used to construct U_e

        pert_coeff :float
            Defines the coefficient lambda by which to apply the perturbation

        timestep :float
            Defines t parameter used in calculating alpha

        u_param :float
            Defines U parameter used in calculating alpha

        '''

        #Define basic attributes of circuit (wire labels, dimensionality, circuit device)
        self.qubits = cir(N, M).qubits
        self.readout_qubits = cir(N,M).readout_qubits
        self.N = N
        self.M = M
        self.dev = cir(N, M).create_device('default.qubit')

        #Define parameters of perturbation
        self.pert_coeff = pert_coeff
        self.t = timestep
        self.u = u_param
        
    def multi_R_6_9(self, theta):

        '''Define a multi-controlled RY gate, used in the composition of the U_dis gate.

        Parameters:
        -----------------------------
        theta :float
            Defines the angle passed through the RY gate

        '''

        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[0]])
        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[1]])
        qml.CNOT(wires=[self.qubits[3], self.qubits[2]])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.RY(0 - theta, wires=self.qubits[3])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.RY(theta, wires=self.qubits[3])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.CNOT(wires=[self.qubits[3], self.qubits[2]])
        qml.CNOT(wires=[self.qubits[3], self.qubits[1]])
        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[0]])
        qml.PauliX(wires=self.qubits[3])

    def multi_R_5_10(self, alpha):
        
        '''Define a multi-controlled RY gate, used in the composition of the U_dis gate.

        Parameters:
        -----------------------------
        alpha :float
            Defines the angle passed through the RY gate

        '''

        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[1]])
        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[0]])
        qml.CNOT(wires=[self.qubits[3], self.qubits[2]])
        qml.PauliX(wires=self.qubits[3])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.RY(0 - alpha, wires=self.qubits[3])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.RY(alpha, wires=self.qubits[3])
        qml.MultiControlledX(wires=[self.qubits[0], self.qubits[1], self.qubits[2], self.qubits[3]])
        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[2]])
        qml.CNOT(wires=[self.qubits[3], self.qubits[0]])
        qml.PauliX(wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[3], self.qubits[1]])
        qml.PauliX(wires=self.qubits[3])

    def qft(self, phase):
        
        '''Define a Quantum Fourier Transform gate that operates on qubits q1 and q2, and q3 and q4

        Parameters:
        -------------------------
        phase :float
            Parametrizes shift in phase

        '''

        #QFT gate applied between q1 and q2:
        qml.PhaseShift(phase, wires=self.qubits[0])
        qml.CNOT(wires=[self.qubits[0], self.qubits[1]])
        qml.ctrl(qml.Hadamard, control=self.qubits[1])(wires=self.qubits[0])
        qml.CNOT(wires=[self.qubits[0], self.qubits[1]])
        qml.ctrl(qml.PauliZ, control=self.qubits[0])(wires=self.qubits[1])

        #QFT gate applied between q3 and q4:
        qml.PhaseShift(phase, wires=self.qubits[2])
        qml.CNOT(wires=[self.qubits[2], self.qubits[3]])
        qml.ctrl(qml.Hadamard, control=self.qubits[3])(wires=self.qubits[2])
        qml.CNOT(wires=[self.qubits[2], self.qubits[3]])
        qml.ctrl(qml.PauliZ, control=self.qubits[2])(wires=self.qubits[3])

    def exp_perturbation(self):
        
        '''Applies the exponential of the desired perturbation to the system qubits
        '''

        #Apply perturbation to qubits q1 and q2 (representative of particle 1):
        qml.CNOT(wires=[self.qubits[0], self.qubits[1]])
        qml.RZ(0 - self.pert_coeff, wires=self.qubits[1])
        qml.CNOT(wires=[self.qubits[0], self.qubits[1]])
        qml.CNOT(wires=[self.qubits[2], self.qubits[1]])
        qml.RZ(0 - self.pert_coeff, wires=self.qubits[1])
        qml.CNOT(wires=[self.qubits[2], self.qubits[1]])

        #Apply perturbation to qubits q3 and q4 (representative of particle 2):
        qml.CNOT(wires=[self.qubits[0], self.qubits[3]])
        qml.RZ(0 - self.pert_coeff, wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[0], self.qubits[3]])
        qml.CNOT(wires=[self.qubits[2], self.qubits[3]])
        qml.RZ(0 - self.pert_coeff, wires=self.qubits[3])
        qml.CNOT(wires=[self.qubits[2], self.qubits[3]])

    def perturbation_check(self):

        '''Undo the perturbation based on result of readout qubit
        '''

        #Initialize readout qubit (uses q2'' for readout):
        qml.PauliX(wires="q2''")
        qml.Hadamard(wires="q2''")

        #Conditional undo of perturbation on qubits q1 and q2:
        qml.Toffoli(wires=[self.qubits[0], "q2''", self.qubits[1]])
        qml.ctrl(qml.RZ, control="q2''")(2*self.pert_coeff, wires=self.qubits[1])
        qml.Toffoli(wires=[self.qubits[0], "q2''", self.qubits[1]])
        qml.Toffoli(wires=[self.qubits[2], "q2''", self.qubits[1]])
        qml.ctrl(qml.RZ, control="q2''")(2*self.pert_coeff, wires=self.qubits[1])
        qml.Toffoli(wires=[self.qubits[2], "q2''", self.qubits[1]])

        #Conditional undo of perturbation on qubits q3 and q4:
        qml.Toffoli(wires=[self.qubits[0], "q2''", self.qubits[3]])
        qml.ctrl(qml.RZ, control="q2''")(2*self.pert_coeff, wires=self.qubits[3])
        qml.Toffoli(wires=[self.qubits[0], "q2''", self.qubits[3]])
        qml.Toffoli(wires=[self.qubits[2], "q2''", self.qubits[3]])
        qml.ctrl(qml.RZ, control="q2''")(2*self.pert_coeff, wires=self.qubits[3])
        qml.Toffoli(wires=[self.qubits[2], "q2''", self.qubits[3]])

        #Convert readout qubit back to computational basis:
        qml.Hadamard(wires="q2''")
        qml.PauliX(wires="q2''")