import numpy as np

def kabsch_rt(A, B):
    centroidA = A.mean(axis=1, keepdims=True)
    centroidB = B.mean(axis=1, keepdims=True)
    A0 = A - centroidA
    B0 = B - centroidB
    H = A0 @ B0.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = centroidB - R @ centroidA
    return R, T
