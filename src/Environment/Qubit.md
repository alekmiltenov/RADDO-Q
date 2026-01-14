# Qubit Properties

- Contains terminologies you can see the meaning of in **[docs/terminology.md](docs/terminology.md)**

### T1 at 0 Kelvin
- A unique behavioural qubit property that can only be measured or estimated
- Energy relaxation (**amplitude damping**) at 0 Kelvin (no thermal excitation, only |1⟩ → |0⟩ decay)

### Gamma1 at 0 Kelvin
- The **relaxation rate** at 0 Kelvin (units: 1/time), i.e. the inverse of T1 at 0K:
  - γ1_0K = 1 / T1_0K
- In a discrete simulation step `dt`, the relaxation probability can be written as:
  - p_down = 1 - e^(-γ1_0K * dt)

### Gamma Phi
- Also a unique behavioural qubit property that represents **pure dephasing** or the qubit's ability to slowly lose its phase (x and y coordinate on the Bloch sphere) even in a perfect environment

### Omega01
- Another qubit property linked to the effect of temperature on the qubit excitation time


### T1 Energy Relaxation Time

- Energy relaxation, also known as **amplitude damping**, is a qubit property that represents the qubit's ability to **lose energy to its environment** (decay from the excited state |1⟩ to the ground state |0⟩).
  - It's a physical limit that cannot be broken
  - In each simulation step `dt`, we calculate probability of a damp happening with:
    - γ = 1 - e^(-dt / T1)
    where:
    - dt is the simulation time step
    - T1 is the measured/predicted energy-relaxation time constant

- Simulation behaves realistically as seen in **[Simulation charts here](Qubit_Coherence_Experiments/Experiments.md)**


### T1 Energy Excitation Time

- Energy excitation or **thermal excitation** represents the qubit's tendency to **gain energy from the environment** (transition from |0⟩ to |1⟩), depending mostly on temperature.
  - In each simulation step `dt`, we often use an excitation probability:
    - γ_up = 1 - e^(-dt / T_up)
    where:
    - dt is the simulation time step
    - T_up is the measured/predicted excitation time constant (often related to temperature)
  
  - If you want to tie it to T1 directly (optional), a common way is to set:
    - γ_down = 1 - e^(-dt / T1)
    - γ_up = (p_th / (1 - p_th)) * γ_down
    where `p_th` is the thermal excited-state population (between 0 and 1).

- Simulation behaves realistically as seen in **[Simulation charts here](Qubit_Coherence_Experiments/Experiments.md)**
