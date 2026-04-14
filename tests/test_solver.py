import pytest
import numpy as np
from PySymmetry.phys.quantum.solver import ExactDiagonalizationSolver
from PySymmetry.phys.quantum.hamiltonian import MatrixHamiltonian


class TestExactDiagonalizationSolver:
    def test_creation(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        assert solver._H is ham

    def test_solve(self):
        H = np.array([[1, 0], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states, energies = solver.solve()
        assert len(states) == 2
        assert len(energies) == 2

    def test_solve_non_hermitian_raises(self):
        H = np.array([[1, 1], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        with pytest.raises(ValueError):
            solver.solve()

    def test_solve_skip_hermitian_check(self):
        H = np.array([[1, 1], [0, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham, check_hermitian=False)
        states, energies = solver.solve()
        assert len(states) == 2

    def test_ground_state(self):
        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        ket, E = solver.ground_state()
        assert np.isclose(E, 1.0)

    def test_excited_states(self):
        H = np.array([[2, 1], [1, 2]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states = solver.excited_states(n=1)
        assert len(states) == 2

    def test_eigenvalues_sorted(self):
        H = np.array([[2, 0], [0, 1]], dtype=complex)
        ham = MatrixHamiltonian(H)
        solver = ExactDiagonalizationSolver(ham)
        states, energies = solver.solve()
        assert energies[0] <= energies[1]
