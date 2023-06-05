from qiskit.circuit import Parameter
from qiskit import execute
import qiskit

from config import *

class QiskitCircuit():
    '''
        Qiskit implementation of a quantum circuit. It contains basic functions, like
        circuit creation, circuit application, etc.
    '''

    def __init__(self):
        '''
            QiskitCircuit constructor.

            Parameters
            ----------
            Nothig, it uses config file

            Returns
            -------
            None, QiskitCircuit object is created
        '''
        self.n_qubits = NUM_QUBITS
        self.circuit  = qiskit.QuantumCircuit(self.n_qubits)
        self.thetas   = {k: Parameter('Theta' + str(k)) for k in range(NUM_LAYERS * self.n_qubits)}
        self.backend  = SIMULATOR
        self.shots    = NUM_SHOTS

        self.__build_circuit()

    def __build_circuit(self):
        '''
            Implements a N qubit M layer real amplitudes circuit.

                         ###############################
                         #                             # 
            |0>--H---R1--#--x---x---x-----------Rn+1---#............---Z1
            |0>--H---R2--#--|---|---|---x---x---Rn+2---#............---Z2
            |0>--H---R3--#------|---|---|---|---Rn+3---#............---Z3
            .............#..........|.......|..........#.................
            |0>--H---Rn--#----------|-------|---Rn+n---#............---Zn
                         #            Layer            #  Layer 2
                         ###############################


            Parameters
            ----------
            Nothing, it uses self and config

            Returns
            -------
            Nothing, it updates self.circuit
        
        '''
        
        all_qubits = [i for i in range(self.n_qubits)]
        self.circuit.h(all_qubits)
        self.circuit.barrier()

        for N in range(NUM_LAYERS):
            for k in range(0, self.n_qubits):
                self.circuit.ry(self.thetas[k+ N*self.n_qubits] , k)

            if NUM_LAYERS==1 or N < NUM_LAYERS-1:
                for k in range(0, self.n_qubits):
                    for k2 in range(0, self.n_qubits):
                        if k < k2:
                            self.circuit.cx(k2, k)

        self.circuit.measure_all()

    def N_qubit_expectation_Z(self, counts):
        '''
            This function allows to take the measurements and translate them 
            to Z-expectation values for every single qubit.

            Parameters
            ----------
            - counts: quantum circuit simulation/harware results

            Returns
            -------
            - expects: Z-expectation values for every single qubit
        '''
        expects = np.zeros(NUM_QC_OUTPUTS)
        for k in range(NUM_QC_OUTPUTS):
            key = QC_OUTPUTS[k]
            perc = counts.get(key, 0) / self.shots
            expects[k] = perc
        return expects

    def run(self, i):
        '''
            Run the circuit using backed (can be real hardware or simulator)
            
            Parameters
            ----------
            - i: input tensor

            Returns
            -------
            - Z-expectation values for every single qubit
        '''
        params = i
        job_sim = execute(self.circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=[{self.thetas[k]: params[k].item() for k in range(NUM_LAYERS * NUM_QUBITS)}])
        result_sim = job_sim.result()
        counts = result_sim.get_counts(self.circuit)
        return self.N_qubit_expectation_Z(counts)