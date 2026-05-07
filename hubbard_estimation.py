import pennylane as qml
import numpy as np
import math

from initialize_circuit import InitializeCircuit
from construction_gates import ConstructionGates
from controlled_rotations import ControlledRotations

class HubbardEstimation:

    '''Class to construct the circuit required to estimate the 1st order corrections to the 2-electron Hubbard model
    '''

    def __init__(self, N, M, pert_coeff, timestep, u_param):
        '''Construction

        Parameters:
        -----------------------------
        N   : int
            Describes number of qubits representative a system of size 2^N

        M   : int
            Number of ancillary qubits required to implement the U_e gate

        pert_coeff  : float
            Defines the coefficient lambda by which to apply the perturbation

        timestep    : float
            Defines t parameter used in calculating alpha

        u_param     : float
            Defines U parameter used in calculating alpha
        '''

        self.gates = ConstructionGates(N, M, pert_coeff, timestep, u_param)
        self.rot = ControlledRotations(N, M, pert_coeff, timestep, u_param)

        self.qubits = InitializeCircuit(N, M).qubits
        self.N = N
        self.M = M
        self.pert_coeff = pert_coeff
        self.timestep = timestep
        self.u_param = u_param

    def u_disentangle(self, theta, alpha, phase):
        '''Gate which transforms the system qubits from the computational basis into the energy eigenstates of the unperturbed Hamiltonian.

            .. math::
            U_{dis}^{\\dagger}H^{(0)}U_{dis}=\\sum_{n}E_n\\ket{n}\\bra{n}

        Parameters:
        -----------
        theta   : float
            Rotation parameter used in the multicontrolled R_6_9 gate

        alpha   : float
            Rotation parameter used in the multicontrolled R_5_10 gate. Determined by timestep and u_param variables

        phase   : float
            Phase shift implemented by the single-qubit phase gate in the QFT circuit

        Notes:
        ------

        Alpha determined by U and t such that

        .. math:: \\alpha=-2\\arccos{\\frac{2t+\\sqrt{U^2/4+4t^2}}{\\sqrt{U^2/4+(2t+\\sqrt{U^2/4+4t^2})^2}}}

        '''

        qml.PauliX(wires=self.qubits[0])
        qml.PauliX(wires=self.qubits[2])
        self.gates.multi_R_6_9(theta)
        self.gates.multi_R_5_10(alpha)
        self.gates.qft(phase)

    def perturbation(self):
        '''Applies a dipole-dipole interaction to the unperturbed Hubbard Hamiltonian
        '''
        
        self.gates.exp_perturbation()
        self.gates.perturbation_check()

    def u_inverse_energy(self, theta_gs, theta_neg1, theta_0, theta_1, theta_2, theta_h):
        '''Constructs gate used to calculate inverse of 1st order energy corrections to the unperturbed Hubbard Hamiltonian

        Parameters:
        -----------
        theta_gs    : float
            Parameter determined by E_gs, where .. math:: \\sin(\\frac{\\theta_2 + \\theta_0 + \\theta_{gs}}{2} = 0

        theta_neg1  : float
            Parameter determined by E_-1, where .. math:: \\sin(\\frac{\\theta_{-1} + \\theta_0}{2} = \\frac{C}{E_{gs} - E_{-1}}

        theta_0     : float
            Parameter determined by E_0, where .. math:: \\sin(\\frac{theta_0}{2}) = \\frac{C}{E_{gs} - E_0}

        theta_1     : float
            Parameter determined by E_1, where .. math:: \\sin(\\frac{\\theta_1 + \\theta_0}{2} = \\frac{C}{E_{gs} - E_1}

        theta_2     : float
            Parameter determined by E_2, where .. math:: \\sin(\\frac{\\theta_2 + \\theta_0}{2}) = \\frac{C}{E_{gs} - E_2}

        theta_h     : float
            Parameter determined by E_h, where .. math:: \\sin(\\frac{\\theta_neg1 + \\theta_0 + \\theta_h}{2} = \\frac{C}{E_{gs} - E_{h}}
            
        '''

        self.rot.cr_0(theta_0)
        self.rot.cr_1(theta_2)
        self.rot.cr_2(theta_gs)
        self.rot.cr_3(theta_neg1)
        self.rot.cr_4(theta_neg1)
        self.rot.cr_5(theta_h)
        self.rot.cr_6(theta_1)
        self.rot.cr_7(theta_1)