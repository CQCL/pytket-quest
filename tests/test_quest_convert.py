# Copyright 2020-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from pyquest import Register
from pyquest.initialisations import ClassicalState

from pytket.circuit import Circuit, OpType
from pytket.extensions.quest import tk_to_quest


def test_h() -> None:
    reg = Register(1)
    reg.apply_operator(ClassicalState(state_ind=0))
    circ = Circuit(1).H(0)
    quest_circ = tk_to_quest(circ)
    reg.apply_circuit(quest_circ)

    reg0 = Register(1)
    probability = reg.inner_product(reg0) ** 2
    assert np.isclose(probability, 0.5)


def test_bellpairs() -> None:
    reg = Register(2)
    circ = Circuit(2).H(0).CX(0, 1)
    quest_circ = tk_to_quest(circ)
    reg.apply_circuit(quest_circ)

    reg0 = Register(2)
    reg0.apply_operator(ClassicalState(state_ind=0))
    probability = reg0.inner_product(reg) ** 2
    assert np.isclose(probability, 0.5)

    reg1 = Register(2)
    reg1.apply_operator(ClassicalState(state_ind=1))
    probability = reg1.inner_product(reg) ** 2
    assert np.isclose(probability, 0)

    reg2 = Register(2)
    reg2.apply_operator(ClassicalState(state_ind=2))
    probability = reg2.inner_product(reg) ** 2
    assert np.isclose(probability, 0)

    reg3 = Register(2)
    reg3.apply_operator(ClassicalState(state_ind=3))
    probability = reg3.inner_product(reg) ** 2
    assert np.isclose(probability, 0.5)


def test_rx_rotations() -> None:
    circ = Circuit(1).Rx(0.5, 0)
    quest_circ = tk_to_quest(circ)
    reg = Register(1)
    reg.apply_circuit(quest_circ)
    v = reg[:]
    assert np.isclose(v[0], np.sqrt(0.5))
    assert np.isclose(v[1], -1j * np.sqrt(0.5))


def test_swap() -> None:
    circ = Circuit(2).X(1).SWAP(0, 1)
    circ_unitary = circ.get_unitary()

    quest_circ = tk_to_quest(circ)
    quest_circ_unitary = quest_circ.as_matrix(num_qubits=2)
    reg = Register(2)
    reg.apply_circuit(quest_circ)
    assert np.allclose(circ_unitary, quest_circ_unitary)
    assert np.allclose(reg[:], [0, 0, 1, 0])


@pytest.mark.parametrize(
    "gate_params",
    [
        (OpType.U1, 0.19, 0),
        (OpType.U2, [0.19, 0.24], 1),
        (OpType.U3, [0.19, 0.24, 0.3], 2),
    ],
)
def test_ibm_gateset_error(gate_params: tuple[OpType, list[int], int]) -> None:
    circ = Circuit(3)
    op_type, angles, qubit = gate_params
    circ.add_gate(op_type, angles, [qubit])
    with pytest.raises(NotImplementedError):
        tk_to_quest(circ)


def test_control_pauliz() -> None:
    circ = Circuit(2)
    circ.H(0).CZ(0, 1).H(1)
    circ_unitary = circ.get_unitary()

    quest_circ = tk_to_quest(circ)
    quest_circ_unitary = quest_circ.as_matrix(num_qubits=2)
    assert np.allclose(circ_unitary, quest_circ_unitary)
