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

import math

import numpy as np
from pytket.circuit import Circuit, BasisOrder, OpType
from pytket.passes import CliffordSimp

from pytket.extensions.quest import QuESTBackend

PARAM = -0.11176849
backends = [
    QuESTBackend(result_type="state_vector"),
    QuESTBackend(result_type="density_matrix"),
]


def h2_1q_circ(theta: float) -> Circuit:
    circ = Circuit(1)
    circ.Ry(-2 / np.pi * -theta, 0)
    return circ


def h2_2q_circ(theta: float) -> Circuit:
    circ = Circuit(2).X(0)
    circ.Rx(0.5, 0).H(1)
    circ.CX(0, 1)
    circ.Rz((-2 / np.pi) * theta, 1)
    circ.CX(0, 1)
    circ.Rx(-0.5, 0).H(1)
    return circ


def h2_3q_circ(theta: float) -> Circuit:
    circ = Circuit(3).X(0).X(1)
    circ.Rx(0.5, 0).H(1).H(2)
    circ.CX(0, 1).CX(1, 2)
    circ.Rz((-2 / np.pi) * theta, 2)
    circ.CX(1, 2).CX(0, 1)
    circ.Rx(-0.5, 0).H(1).H(2)
    return circ


def h2_4q_circ(theta: float) -> Circuit:
    circ = Circuit(4).X(0).X(1)
    circ.Rx(0.5, 0).H(1).H(2).H(3)
    circ.CX(0, 1).CX(1, 2).CX(2, 3)
    circ.Rz((-2 / np.pi) * theta, 3)
    circ.CX(2, 3).CX(1, 2).CX(0, 1)
    circ.Rx(-0.5, 0).H(1).H(2).H(3)
    return circ


def test_properties() -> None:
    svb = QuESTBackend()
    dmb = QuESTBackend(result_type="density_matrix")
    assert not svb._density_matrix
    assert dmb._density_matrix


def test_get_state() -> None:
    quest_circ = h2_4q_circ(PARAM)
    correct_state = np.array(
        [
            -4.97881051e-19 + 3.95546482e-17j,
            -2.04691245e-17 + 4.26119488e-18j,
            -2.05107665e-17 - 1.16628720e-17j,
            -1.11535930e-01 - 2.20309881e-16j,
            1.14532773e-16 + 1.84639112e-16j,
            -2.35945152e-18 + 1.00839027e-17j,
            -3.27177146e-18 - 1.35977120e-17j,
            1.68171141e-17 - 3.67997979e-17j,
            6.96542384e-18 + 6.20603820e-17j,
            2.94777720e-17 + 1.82756571e-19j,
            1.43716480e-17 + 3.62382653e-18j,
            3.41937038e-17 - 8.77511869e-18j,
            9.93760402e-01 + 1.59594560e-15j,
            -2.73151084e-18 + 6.31487294e-17j,
            2.09501038e-17 + 6.22364095e-17j,
            -8.59510231e-18 + 5.90202794e-18j,
        ]
    )
    for b in backends:
        quest_circ = b.get_compiled_circuit(quest_circ)
        if b.supports_state:
            quest_state = b.run_circuit(quest_circ).get_state()
            assert np.allclose(quest_state, correct_state)
        if b.supports_density_matrix:
            quest_state = b.run_circuit(quest_circ).get_density_matrix()
            assert np.allclose(
                quest_state, np.outer(correct_state, correct_state.conj())
            )


def test_statevector_phase() -> None:
    for b in backends:
        if not b.supports_state:
            continue
        circ = Circuit(2)
        circ.H(0).CX(0, 1)
        circ = b.get_compiled_circuit(circ)
        state = b.run_circuit(circ).get_state()
        assert np.allclose(state, [math.sqrt(0.5), 0, 0, math.sqrt(0.5)], atol=1e-10)
        circ.add_phase(0.5)
        state1 = b.run_circuit(circ).get_state()
        assert np.allclose(state1, state * 1j, atol=1e-10)


def test_swaps_basisorder() -> None:
    # Check that implicit swaps can be corrected irrespective of BasisOrder
    for b in backends:
        c = Circuit(4)
        c.X(0)
        c.CX(0, 1)
        c.CX(1, 0)
        CliffordSimp(True).apply(c)
        assert c.n_gates_of_type(OpType.CX) == 1
        c = b.get_compiled_circuit(c)
        res = b.run_circuit(c)
        if b.supports_state:
            s_ilo = res.get_state(basis=BasisOrder.ilo)
            s_dlo = res.get_state(basis=BasisOrder.dlo)
            correct_ilo = np.zeros((16,))
            correct_ilo[4] = 1.0
            assert np.allclose(s_ilo, correct_ilo)
            correct_dlo = np.zeros((16,))
            correct_dlo[2] = 1.0
            assert np.allclose(s_dlo, correct_dlo)
        if b.supports_density_matrix:
            s_ilo = res.get_density_matrix(basis=BasisOrder.ilo)
            s_dlo = res.get_density_matrix(basis=BasisOrder.dlo)
            correct_ilo = np.zeros((16,))
            correct_ilo[4] = 1.0
            assert np.allclose(s_ilo, np.outer(correct_ilo, correct_ilo.conj()))
            correct_dlo = np.zeros((16,))
            correct_dlo[2] = 1.0
            assert np.allclose(s_dlo, np.outer(correct_dlo, correct_dlo.conj()))


def test_default_pass() -> None:
    for b in backends:
        for ol in range(3):
            comp_pass = b.default_compilation_pass(ol)
            c = Circuit(3, 3)
            c.H(0)
            c.CX(0, 1)
            c.CSWAP(1, 0, 2)
            c.ZZPhase(0.84, 2, 0)
            c.measure_all()
            comp_pass.apply(c)
            for pred in b.required_predicates:
                assert pred.verify(c)


def test_backend_info() -> None:
    for b in backends:
        assert b.backend_info is not None
