"""PySymmetry 工具模块

该模块提供各种实用工具，包括：
- 符号计算工具（基于 SymPy）
- 性能优化工具
"""

from .lie_algebra_symbolic import (
    SymbolicLieAlgebra,
    SymbolicLieAlgebraElement,
    SymbolicLieBracket,
    SymbolicKillingForm,
    SymbolicWeylGroup,
    compute_structure_constants,
    verify_jacobi_identity,
    generate_weyl_coordinates,
)

from .performance import (
    cache_result,
    vectorize_func,
    optimize_matrix_multiply,
    batch_matrix_multiply,
    optimize_eigendecomposition,
    parallel_apply,
    optimize_kron_sequence,
    optimize_wigner_d,
    optimize_blevit_check,
    sparse_diagonalize,
    optimize_trace,
    optimize_outer_sequence,
    matrix_power_sequence,
    block_diagonalize,
    optimize_commutator,
    optimize_anticommutator,
    batch_evaluate,
    NUMBA_AVAILABLE,
    JOBLIB_AVAILABLE,
)

__all__ = [
    'SymbolicLieAlgebra',
    'SymbolicLieAlgebraElement',
    'SymbolicLieBracket',
    'SymbolicKillingForm',
    'SymbolicWeylGroup',
    'compute_structure_constants',
    'verify_jacobi_identity',
    'generate_weyl_coordinates',
    'cache_result',
    'vectorize_func',
    'optimize_matrix_multiply',
    'batch_matrix_multiply',
    'optimize_eigendecomposition',
    'parallel_apply',
    'optimize_kron_sequence',
    'optimize_wigner_d',
    'optimize_blevit_check',
    'sparse_diagonalize',
    'optimize_trace',
    'optimize_outer_sequence',
    'matrix_power_sequence',
    'block_diagonalize',
    'optimize_commutator',
    'optimize_anticommutator',
    'batch_evaluate',
    'NUMBA_AVAILABLE',
    'JOBLIB_AVAILABLE',
]