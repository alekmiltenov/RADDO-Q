import numpy as np

np.set_printoptions(precision=3, suppress=True)


def apply_Unitary(rho: np.ndarray , U: np.ndarray) -> np.ndarray:
    rho = U @ rho @ U.conj().T
    return rho


def q_init():
    rho = np.array([[1,0],
                    [0,0]], dtype=np.complex128)
    return rho

def q_excite():
    rho = np.array([[0,0],
                    [0,1]], dtype=np.complex128)
    return rho


def q_Rx(theta, rho):
    cos = np.cos(theta / 2.0)
    sin = np.sin(theta / 2.0)
    U = np.array([[cos , -1j * sin],
                  [-1j * sin , cos]], dtype=np.complex128)

    rho = apply_Unitary(rho, U)
    return rho


def q_Ry(theta, rho):
    cos = np.cos(theta / 2.0)
    sin = np.sin(theta / 2.0)
    U = np.array([[cos , -sin],
                  [sin , cos]], dtype=np.complex128)

    rho = apply_Unitary(rho, U)
    return rho


def q_Rz(theta, rho):
    e_j     = np.exp(+1j * (theta / 2.0))
    e_neg_j = np.exp(-1j * (theta / 2.0))
    U = np.array([[e_neg_j , 0.0],
                  [0.0 , e_j]], dtype=np.complex128)
    
    rho = apply_Unitary(rho, U)
    return rho
