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

    # Avoid Unneccessary Compute helpers
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
            self.temperature_Kelvin = 0.001  
        else :
            self.temperature_Kelvin = float(temperature_Kelvin)
        self.omega_01 = float(omega_01)

        self.gamma_down = self._gamma_down(self.temperature_Kelvin)
        self.gamma_up = self._gamma_up(self.temperature_Kelvin, omega_01, self.gamma_down)

        self.base_relaxation_probability =   (1 - np.exp( - self.gamma_down * self.dt))
        self.base_excitation_probability =   (1 - np.exp( - self.gamma_up   * self.dt))
        self.rng = np.random.default_rng(seed=42)


    @staticmethod
    def _gamma_down(temperature_Kelvin: float) -> float:
        if temperature_Kelvin <= 0.0 : 
            temperature_Kelvin = 0.001                                  # 1 mK
        A = 2.28e3                                                      # Orbach
        B = 2.5e-11                                                     # Raman
        delta_kelvin = 875.0                                            # delta kelvin = Activation energy E / Boltzmann constant kB    Joules to Kelvin Precompute

        gamma_down = B * (temperature_Kelvin**5) + A * np.exp(-(delta_kelvin / temperature_Kelvin))
        return gamma_down

    @staticmethod
    def _gamma_up(temperature_Kelvin: float, omega_01: float, gamma_down: float) -> float:
        if temperature_Kelvin <= 0.0 : 
            temperature_Kelvin = 0.001                                  # 1 mK

        kB = 1.380649e-23                                               # Boltzmann Constant
        hbar = 1.054571817e-34                                          # Reduced Planck Constant

        gamma_up = gamma_down * np.exp( -(hbar * omega_01) / (kB * temperature_Kelvin) )
        return gamma_up


    # Probability of relaxation
    def relaxation_probability(self) -> float:
        rho11 = float(np.real(self.rho[1, 1]))
        rho11 = min(max(rho11, 0.0), 1.0)
        P_relaxation_jump = rho11 * self.base_relaxation_probability
        return P_relaxation_jump


    # Probability of excitation
    def excitation_probability(self) -> float:
        rho00 = float(np.real(self.rho[0, 0]))
        rho00 = min(max(rho00, 0.0), 1.0)

        P_excitation_jump = rho00 * self.base_excitation_probability
        return P_excitation_jump
    
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


    # Generalized Amplitude Damping - decide if jump happens
    def GAD(self):
        random = self.rng.random()

        P_down = self.relaxation_probability()
        P_up = self.excitation_probability()

        if random < P_down : 
            self.rho = self.apply_relaxation_jump()
            print("Relaxation jump occured")
            return

        if random < P_down + P_up : 
            self.rho = self.apply_excitation_jump()
            return
        
        self.rho = self.apply_GAD_nojump(self.rho, self.base_relaxation_probability , self.base_excitation_probability)