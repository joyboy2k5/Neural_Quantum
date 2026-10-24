
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a circuit with one qubit
circuit = QuantumCircuit(1)

# Put the qubit into a superposition
circuit.h(0)

# Display the circuit
print(circuit.draw())

# Simulate the quantum state
state = Statevector.from_instruction(circuit)

print("\nQuantum state:")
print(state)
print("\nMeasurement probabilities:")
print(state.probabilities_dict())
