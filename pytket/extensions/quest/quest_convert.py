# Copyright 2019-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
##
#     http://www.apache.org/licenses/LICENSE-2.0
##
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion from to tket circuits to QuEST circuits
"""
import numpy as np

import pyquest.unitaries as gates
from pyquest.gates import M as Measurement
from pyquest import Circuit as PyQuESTCircuit

from pytket.circuit import Circuit, OpType
from pytket.passes import FlattenRegisters


_ONE_QUBIT_GATES = {
    OpType.X: gates.NOT,
    OpType.Y: gates.Y,
    OpType.Z: gates.Z,
    OpType.H: gates.H,
    OpType.S: gates.S,
    OpType.T: gates.T,
}

_ONE_QUBIT_ROTATIONS = {OpType.Rx: gates.Rx, OpType.Ry: gates.Ry, OpType.Rz: gates.Rz}

_MEASURE_GATES = {OpType.Measure: Measurement}

_TWO_QUBIT_GATES = {OpType.CX: gates.X, OpType.CZ: gates.Z, OpType.SWAP: gates.Swap}


def tk_to_quest(
    circuit: Circuit, reverse_index: bool = False, replace_implicit_swaps: bool = False
) -> PyQuESTCircuit:
    """Convert a pytket circuit to a quest circuit object."""
    circ = circuit.copy()

    if not circ.is_simple:
        FlattenRegisters().apply(circ)

    if replace_implicit_swaps:
        circ.replace_implicit_wire_swaps()
    n_qubits = circ.n_qubits
    quest_operators = []
    index_map = {
        i: (i if not reverse_index else n_qubits - 1 - i) for i in range(n_qubits)
    }
    for com in circ:
        optype = com.op.type
        if optype in _ONE_QUBIT_GATES:
            quest_gate = _ONE_QUBIT_GATES[optype]
            index = index_map[com.qubits[0].index[0]]
            add_gate = quest_gate(index)
            quest_operators.append(add_gate)

        elif optype in _ONE_QUBIT_ROTATIONS:
            quest_gate = _ONE_QUBIT_ROTATIONS[optype]
            index = index_map[com.qubits[0].index[0]]
            param = com.op.params[0] * np.pi  # type: ignore
            add_gate = quest_gate(index, param)
            quest_operators.append(add_gate)

        elif optype in _TWO_QUBIT_GATES:
            id1 = index_map[com.qubits[0].index[0]]
            id2 = index_map[com.qubits[1].index[0]]
            quest_gate = _TWO_QUBIT_GATES[optype]
            if optype == OpType.SWAP:
                add_gate = quest_gate(targets=[id1, id2])
            elif optype == OpType.CZ:
                add_gate = gates.Z(id2, controls=id1)
            else:
                add_gate = quest_gate(id2, controls=id1)
            quest_operators.append(add_gate)

        elif optype in _MEASURE_GATES:
            continue

        elif optype == OpType.Barrier:
            continue

        else:
            raise NotImplementedError(
                "Gate: {} Not Implemented in QuEST!".format(optype)
            )

    quest_circ = PyQuESTCircuit(quest_operators)
    return quest_circ
