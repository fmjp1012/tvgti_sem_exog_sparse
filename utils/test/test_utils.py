import numpy as np
import pytest
from utils import (
    elimination_matrix_h,
    duplication_matrix_h,
    elimination_matrix_hh,
    duplication_matrix_hh,
    soft_thresholding,
    project_to_zero_diagonal_symmetric,
)


def test_smoke_elimination_duplication_roundtrip():
    N = 5
    S = np.random.rand(N, N)
    S = (S + S.T) / 2
    E = elimination_matrix_h(N).tocsc()
    D = duplication_matrix_h(N).tocsc()
    vech_S = E.dot(S.flatten())
    vec_S = D.dot(vech_S)
    S2 = vec_S.reshape(N, N)
    np.testing.assert_array_almost_equal(S2, S)


def test_smoke_project_to_zero_diagonal_symmetric():
    A = np.random.rand(6, 6)
    P = project_to_zero_diagonal_symmetric(A)
    assert np.allclose(P, P.T)
    assert np.allclose(np.diag(P), 0.0)


