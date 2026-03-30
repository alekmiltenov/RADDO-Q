import numpy as np

class Qubit:
    dt : float                      # Time step in simulation

    # Qubit Properties
    rho: np.ndarray                 # Qubt as Density Matrix
    gamma_down: float               # Relaxation rate ( 1 / T1 )
    gamma_up: float                 # Excitation rate
    omega_01: float                 # Qubit Transiition Frequency rad/s
    temperature_Kelvin: float       # Environment Temperature in Kelvin

    # Qubit Measurements
    T1: float
    T2_star: float
    T2: float
    T2_eff: float
    T2_eff_agent: float

    # Avoid Unnecessary Compute helpers
    base_relaxation_probability: float
    base_excitation_probability: float
    rng : np.random.Generator


    def __init__(self, rho, dt = 1e-5, omega_01 = 1.8036e10, temperature_Kelvin=77.0):
        if dt <= 0.0:
            self.dt = 1e-5
        else :
            self.dt = float(dt)

        self.rho = rho.astype(np.complex128)

        if temperature_Kelvin <= 0.0 : 
            self.temperature_Kelvin = 77.0  
        else :
            self.temperature_Kelvin = float(temperature_Kelvin)
        self.omega_01 = float(omega_01)

        self.gamma_down = self._gamma_down(self.temperature_Kelvin)
        self.gamma_up = self._gamma_up(self.temperature_Kelvin, omega_01, self.gamma_down)

        self.base_relaxation_probability =   (1 - np.exp( - self.gamma_down * self.dt))
        self.base_excitation_probability =   (1 - np.exp( - self.gamma_up   * self.dt))
        self.rng = np.random.default_rng()


    @staticmethod
    def _gamma_down(temperature_Kelvin: float) -> float:
        if temperature_Kelvin <= 0.0 : 
            temperature_Kelvin = 77.0                                   # 77 K
        A = 2.28e3                                                      # Orbach
        B = 2.5e-11                                                     # Raman
        delta_kelvin = 875.0                                            # delta kelvin = Activation energy E / Boltzmann constant kB    Joules to Kelvin Precompute

        return (B * (temperature_Kelvin**5) + A * np.exp(-(delta_kelvin / temperature_Kelvin)))

    @staticmethod
    def _gamma_up(temperature_Kelvin: float, omega_01: float, gamma_down: float) -> float:
        if temperature_Kelvin <= 0.0 : 
            temperature_Kelvin = 77.0                                   # 77 K
        kB = 1.380649e-23                                               # Boltzmann Constant
        hbar = 1.054571817e-34                                          # Reduced Planck Constant

        return (gamma_down * np.exp( -(hbar * omega_01) / (kB * temperature_Kelvin)))


    @staticmethod
    # Probability of relaxation
    def relaxation_probability(rho: np.ndarray, base_relaxation_probability : float) -> float:
        rho11 = float(np.real(rho[1, 1]))
        rho11 = min(max(rho11, 0.0), 1.0)
        return (rho11 * base_relaxation_probability)
    
    # Probability of excitation
    @staticmethod
    def excitation_probability(rho: np.ndarray, base_excitation_probability : float) -> float:
        rho00 = float(np.real(rho[0, 0]))
        rho00 = min(max(rho00, 0.0), 1.0)
        return (rho00 * base_excitation_probability)
    

    # Relaxation Jump
    @staticmethod
    def apply_relaxation_jump() -> np.ndarray:
        return np.array([[1.0, 0.0],
                        [0.0, 0.0]], dtype=np.complex128)

    # Excitation Jump
    @staticmethod
    def apply_excitation_jump() -> np.ndarray:
        return np.array([[0.0, 0.0],
                        [0.0, 1.0]], dtype=np.complex128)


    # GAD no jump
    @staticmethod
    def apply_GAD_nojump(rho: np.ndarray, P_down_unweighted: float, P_up_unweighted: float) -> np.ndarray:
        P_down_unweighted = min( max(float(P_down_unweighted), 0.0), 1.0)
        P_up_unweighted   = min( max(float(P_up_unweighted)  , 0.0), 1.0)

        K0 = np.array([ [ np.sqrt(1.0 - P_up_unweighted) ,               0.0               ] ,
                        [               0.0              , np.sqrt(1.0 - P_down_unweighted)]
                      ], dtype=np.complex128)

        rho_nojump = K0 @ rho @ K0.conj().T
        tr = float(np.real(np.trace(rho_nojump)))
        if tr > 0.0:
            rho_nojump = rho_nojump / tr
        return rho_nojump


    # Generalized Amplitude Damping
    def GAD(self):
        random = self.rng.random()

        P_down = self.relaxation_probability(self.rho, self.base_relaxation_probability) # could skip some self accessing
        P_up = self.excitation_probability(self.rho, self.base_excitation_probability) # could skip self

        if random < P_down : 
            self.rho = self.apply_relaxation_jump()
            # print("Relaxation jump occured")
            return 0

        if random < P_down + P_up : 
            self.rho = self.apply_excitation_jump()
            # print("Excitation jump occured")
            return 1
        
        self.rho = self.apply_GAD_nojump(self.rho, self.base_relaxation_probability , self.base_excitation_probability)
        return -1
        #####! could pass in rho to skip self...



def main():
    from qubit_gates import q_Ry
    rho = np.array([[0.0, 0.0],
                     [0.0, 1.0]], dtype=np.complex128)
    rho = q_Ry(np.pi/2, rho)

    qubit = Qubit(rho=rho, dt=1e-5, temperature_Kelvin=77.0)

    print("gamma_down:", qubit.gamma_down, "T1:", 1/qubit.gamma_down)
    print("gamma_up:", qubit.gamma_up, "ratio up/down:", qubit.gamma_up/qubit.gamma_down)
    print("p_down:", qubit.base_relaxation_probability)
    print("p_up:", qubit.base_excitation_probability)
    print("rho11 start:", float(np.real(qubit.rho[1,1])))
    print("rho00 start:", float(np.real(qubit.rho[0,0])))

    t = 0.0
    last_print = 0.0
    print_interval = 0.001

    while t < 24.0:
        qubit.GAD()
        t += qubit.dt

        if t - last_print >= print_interval:
            rho11 = float(np.real(qubit.rho[1, 1]))
            print(f"t = {t:.4f}s | rho11 = {rho11:.6f} | rho00 = {1-rho11:.6f}")
            last_print = t


if __name__ == "__main__":
    main()
    ###!!! avoid self accessing