import numpy as np
from qubit import Qubit

rho = np.zeros((2,2), dtype=np.complex128)
np.set_printoptions(precision=3, suppress=True)


def apply_Unitary(rho: np.ndarray , U: np.ndarray) -> np.ndarray:
    rho = U @ rho @ U.conj().T
    return rho


def q_init(rho):
    rho = np.array([[1,0],
                    [0,0]], dtype=np.complex128)
    return rho

def q_excite(rho):
    rho = q_Rx(np.pi, rho)
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


def main():
    rho = np.zeros((2,2), dtype=np.complex128)
    print("|0>")
    rho = q_init(rho)
    print(rho)
    print("\n")

    print("|1>")
    rho = q_Rx(np.pi,rho)
    print(rho)
    print("\n")

    print("|0>")
    rho = q_Ry(np.pi,rho)
    print(rho)
    print("\n")

    print("|0>")
    rho = q_Rz(np.pi,rho)
    print(rho)
    print("\n")

if __name__ == "__main__":
    main()