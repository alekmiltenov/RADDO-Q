# RADDO-Q  
**Reinforcement-Driven Adaptive Dynamic Decoupling for Qubit Coherence Optimization**

---

## Overview

RADDO-Q is a research-driven project focused on extending qubit coherence time (T₂) through intelligent design of dynamical decoupling (DD) pulse sequences.

Instead of relying purely on handcrafted sequences like CPMG or XY, the project explores whether reinforcement learning (RL) can discover pulse patterns that perform better under realistic, noisy quantum environments.

The core idea is simple:
Treat pulse sequence design as an optimization problem and let an agent learn what works best.

---

## Problem

Quantum systems lose information due to environmental noise, including:
- Dephasing (T₂)
- Energy relaxation (T₁)
- 1/f noise
- Random telegraph noise (RTN)
- Control imperfections

Traditional dynamical decoupling sequences:
- Are analytically designed
- Assume simplified noise models
- Do not adapt to real hardware conditions

The goal of RADDO-Q is to find pulse sequences that maximize coherence under realistic, mixed-noise environments.

---

## Approach

RADDO-Q combines three core components:
- Physics-based qubit simulation
- Realistic noise modeling
- Reinforcement learning

---

### Qubit Simulation

The system is modeled as a two-level quantum system using a density matrix ρ.

- Time evolves in discrete steps (dt)
- Dephasing is applied through phase accumulation
- Relaxation is modeled using Generalized Amplitude Damping (GAD)

---

### Noise Model

The environment includes multiple interacting noise sources:

- Ornstein–Uhlenbeck (OU) noise (slow and fast components)
- Random Telegraph Noise (RTN)
- 1/f noise implemented via a bank of RTN fluctuators
- Quasi-static detuning
- White dephasing noise (γφ)
- Technical imperfections such as:
  - Pulse amplitude errors
  - Timing jitter
  - Phase noise

This creates a highly stochastic and realistic simulation environment.

---

### Dynamical Decoupling

Pulse sequences are applied to counteract noise:

- π pulses around X and Y axes
- Arbitrary sequence patterns
- Variable time spacing (τ) between pulses

Coherence is measured as:
C = 2 * |⟨ρ₀₁⟩|

---

### T₂ Extraction

Coherence decay is analyzed using multiple methods:

- First drop below 1/e
- Envelope-based decay tracking
- Exponential fitting:
  C(t) = A * exp(-t / T₂)

---

### Reinforcement Learning

The problem is framed as a sequence optimization task.

Instead of reacting to internal quantum states (which are not observable in real systems), the agent follows a simple loop:

Generate sequence → Evaluate performance → Learn

---

#### Environment

- The agent generates a full pulse sequence
- The sequence is evaluated through simulation
- Reward is based on achieved T₂

Future direction includes step-wise control, where the agent selects pulses dynamically and receives intermediate rewards based on coherence survival.

---

### Algorithms

- PPO (Proximal Policy Optimization)
- Recurrent PPO (LSTM-based memory)

Planned extensions:
- Transformer-based sequence models
- Attention-based policies for pulse generation

---

## Key Insights

- The agent often rediscovers known sequences like CPMG and XY variants
- It also discovers non-trivial patterns such as:
  - Long Y-dominant sequences with occasional X corrections
  - Block-based structures (e.g., repeated Y segments followed by X)

This suggests that the search space contains meaningful improvements beyond standard DD sequences.

---

## Limitations

- High computational cost (sequence evaluation is expensive)
- Noise model is approximate and not hardware-calibrated
- Time discretization introduces rounding effects
- No direct validation on real quantum hardware yet

---

## Future Work

Short-term:
- Optimize simulation performance
- Improve reward shaping
- Introduce variable τ learning
- Add transformer-based models

Mid-term:
- Learn noise-adaptive pulse strategies
- Reduce reliance on brute-force parameter sweeps

Long-term:
- Integrate with real quantum hardware such as :contentReference[oaicite:0]{index=0}
- Experimentally validate learned sequences
- Investigate axis-dependent noise (X vs Y imbalance)

---

## Tech Stack

- Python
- NumPy / SciPy
- Gymnasium
- Stable-Baselines3
- sb3-contrib (Recurrent PPO)
- Matplotlib
- Joblib

---

## Vision

RADDO-Q aims to move dynamical decoupling from static, human-designed sequences to adaptive, learned strategies optimized for real quantum environments.

Better pulse sequences mean longer coherence times, which directly improves the reliability and scalability of quantum computation.

The central question is:
Can machine learning outperform traditional quantum control design?

RADDO-Q is built to explore that.