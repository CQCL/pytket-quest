# Copyright 2019-2024 Quantinuum
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

"""Methods to allow tket circuits to be ran on the QuEST simulator
"""

from typing import List, Sequence
from pytket.backends import (
    Backend,
    CircuitStatus,
    ResultHandle,
)
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit
from pytket.passes import (
    BasePass,
)
from pytket.predicates import (
    Predicate,
)


class QuESTBackend(Backend):
    """
    Backend for running simulations on the QuEST simulator
    """

    _supports_shots = False
    _supports_counts = False
    _supports_state = True
    _supports_unitary = False
    _supports_density_matrix = True
    _supports_expectation = False
    _expectation_allows_nonhermitian = False
    _supports_contextual_optimisation = False
    _persistent_handles = False

    def __init__(
        self,
        result_type: str = "state_vector",
    ) -> None:
        """
        Backend for running simulations on the QuEST simulator

        :param result_type: Indicating the type of the simulation result
            to be returned. It can be either "state_vector" or "density_matrix".
            Defaults to "state_vector"
        """
        super().__init__()

    @property
    def required_predicates(self) -> List[Predicate]:
        raise NotImplementedError

    def rebase_pass(self) -> BasePass:
        raise NotImplementedError

    def default_compilation_pass(self, optimisation_level: int = 2) -> BasePass:
        raise NotImplementedError

    @property
    def _result_id_type(self, ) -> _ResultIdTuple:
        raise NotImplementedError

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: int | Sequence[int] | None = None,
        valid_check: bool = True,
        **kwargs: int | float | str | None
    ) -> List[ResultHandle]:
        raise NotImplementedError

    def circuit_status(self, handle: ResultHandle) -> CircuitStatus:
        raise NotImplementedError
