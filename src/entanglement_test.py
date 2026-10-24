
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a circuit with two qubits
circuit = QuantumCircuit(2)

# Put the first qubit into superposition
circuit.h(0)

# Entangle the two qubits
circuit.cx(0, 1)

print(circuit.draw())

state = Statevector.from_instruction(circuit)

print("\nQuantum state:")
print(state)

print("\nMeasurement probabilities:")
print(state.probabilities_dict())
