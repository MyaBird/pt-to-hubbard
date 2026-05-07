import pennylane as qml
import numpy as np
import math

from initialize_circuit import InitializeCircuit

class ControlledRotations:

    '''Define a series of multi-controlled RY gates used in the construction of U_e. 
    The parameters for each rotation (theta_n) are determined by the corresponding energy levels of the unperturbed Hamiltonian.
    '''
    
    def __init__(self, N, M, pert_coeff, timestep, u_param):

        '''Construction

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

        cir = InitializeCircuit(N,M)

        #Define basic attributes of circuit (wire labels, dimensionality, circuit device)
        self.qubits = cir.qubits
        self.readout_qubits = cir.readout_qubits
        self.N = N
        self.M = M
        self.dev = cir.create_device('default.qubit')

        #Define parameters of perturbation
        self.pert_coeff = pert_coeff
        self.t = timestep
        self.u = u_param

        #Define parameter angle used in multicontrolled RY_5_10 gate (defined below)
        #self.alpha = -2 * math.acos((2*self.t + np.sqrt(((self.u ** 2) / 4) + (4 * (self.t ** 2)))) / (np.sqrt(((self.u ** 2) / 4) + (2*self.t + np.sqrt(((self.u ** 2) / 4) + (4 * (self.t ** 2))) ** 2))))

    def cr_0(self, theta_0):
        '''Sets energy state associated with E_0 as the default value on q1''

        Parameters:
        -------------------------
        theta_0 : int
            Parameter determined by E_0, where .. math:: \\sin(\\frac{theta_0}{2}) = \\frac{C}{E_{gs} - E_0}

        '''

        qml.RY(theta_0, wires="q1''")

    def cr_1(self, theta_2):
        '''Correction term to account for E_2

        Parameters:
        -------------------------
        theta_2 : int
            Parameter determined by E_2, where .. math:: \\sin(\\frac{\\theta_2 + \\theta_0}{2}) = \\frac{C}{E_{gs} - E_2}

        '''

        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[2])
        qml.RY(0 - (theta_2 / 2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1''"])
        qml.RY(theta_2 / 2, wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1''"])
        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[2])

    def cr_2(self, theta_gs):
        '''Correction term to account for E_gs (ground state energy)

        Parameters:
        -------------------------
        theta_gs :int
            Parameter determined by E_gs, where .. math:: \\sin(\\frac{\\theta_2 + \\theta_0 + \\theta_{gs}}{2} = 0

        '''

        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[2])
        qml.PauliX(wires=self.qubits[3])
        qml.RY(0 - (theta_gs / 2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[1], "q1'"])
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.RY(theta_gs / 2, wires="q1''")
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.Toffoli(wires=[self.qubits[0], self.qubits[1], "q1'"])
        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[2])
        qml.PauliX(wires=self.qubits[3])

    def cr_3(self, theta_neg1):
        '''1st correction term to account for E_-1

        Parameters:
        -------------------------
        theta_neg1 :int
            Parameter determined by E_-1, where .. math:: \\sin(\\frac{\\theta_{-1} + \\theta_0}{2} = \\frac{C}{E_{gs} - E_{-1}}

        '''

        qml.RY(0 - (theta_neg1/2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1''"])
        qml.RY(theta_neg1/2, wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1''"])

    def cr_4(self, theta_neg1):
        '''2nd correction term to account for E_-1

        Parameters:
        -------------------------
        theta_neg1 :int
            Parameter determined by E_-1, where .. math:: \\sin(\\frac{\\theta_{-1} + \\theta_0}{2} = \\frac{C}{E_{gs} - E_{-1}}

        '''

        qml.RY(theta_neg1/2, wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1'"])
        qml.Toffoli(wires=[self.qubits[1], self.qubits[3], "q2'"])
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.RY(0 - (theta_neg1/2), wires="q1''")
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.Toffoli(wires=[self.qubits[1], self.qubits[3], "q2'"])
        qml.Toffoli(wires=[self.qubits[0], self.qubits[2], "q1'"])

    def cr_5(self, theta_h):
        '''Correction term to account for E_h (highest energy level)

        Parameters:
        -------------------------
        theta_h :int
            Parameter determined by E_h, where .. math:: \\sin(\\frac{\\theta_neg1 + \\theta_0 + \\theta_h}{2} = \\frac{C}{E_{gs} - E_{h}}

        '''

        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[3])
        qml.RY(0 - (theta_h/2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[1], self.qubits[3], "q2'"])
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.RY(theta_h/2, wires="q1''")
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.Toffoli(wires=[self.qubits[1], self.qubits[3], "q2'"])
        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[3])

    def cr_6(self, theta_1):
        '''1st correction term to account for E_1

        Parameters:
        -------------------------
        theta_1 :int
            Parameter determined by E_1, where .. math:: \\sin(\\frac{\\theta_1 + \\theta_0}{2} = \\frac{C}{E_{gs} - E_1}

        '''

        qml.PauliX(wires=self.qubits[2])
        qml.PauliX(wires=self.qubits[3])
        qml.RY(0 - (theta_1/2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.Toffoli(wires=[self.qubits[0], "q2'", "q1''"])
        qml.RY(theta_1/2, wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], "q2'", "q1''"])
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.PauliX(wires=self.qubits[2])
        qml.PauliX(wires=self.qubits[3])

    def cr_7(self, theta_1):
        '''2nd correction term to account for E_1

        Parameters:
        -------------------------
        theta_1 :int
            Parameter determined by E_1, where .. math:: \\sin(\\frac{\\theta_1 + \\theta_0}{2} = \\frac{C}{E_{gs} - E_1}

        '''

        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[3])
        qml.RY(0 - (theta_1/2), wires="q1''")
        qml.Toffoli(wires=[self.qubits[0], self.qubits[1], "q1'"])
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.RY(theta_1/2, wires="q1''")
        qml.Toffoli(wires=["q1'", "q2'", "q1''"])
        qml.Toffoli(wires=[self.qubits[2], self.qubits[3], "q2'"])
        qml.Toffoli(wires=[self.qubits[0], self.qubits[1], "q1'"])
        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[1])
        qml.PauliX(wires=self.qubits[3])