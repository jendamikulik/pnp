import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Helper: random unitary matrix (U^† U = I)
# ------------------------------------------------------------
def random_unitary(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Complex random matrix
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    # QR decomposition gives Q unitary (up to phases)
    Q, R = np.linalg.qr(A)
    # Fix phases on the diagonal of R so that Q is strictly unitary
    diag_phases = np.diag(R) / np.abs(np.diag(R))
    U = Q * diag_phases
    return U

# ------------------------------------------------------------
# 2) Build "quantum foam": lattice + complex field ψ(x)
# ------------------------------------------------------------
def build_field_and_kernel(N=128, L=10.0, sigma=1.0, k_phase=3.0, ell=1.5):
    """
    N      ... number of lattice points (foam bubbles)
    L      ... length of the interval [0, L]
    sigma  ... width of amplitude envelope
    k_phase... linear phase ramp (wave vector)
    ell    ... correlation length in kernel
    """
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    # Complex field ψ(x) = A(x) * exp(i φ(x))
    A = np.exp(-0.5 * ((x - L/2) / sigma) ** 2)           # Gaussian envelope
    phi = k_phase * x                                     # linear phase
    psi = A * np.exp(1j * phi)                            # complex state vector

    # Kernel: real symmetric Gaussian coupling
    # K_ij = exp( - (x_i - x_j)^2 / (2 ell^2) )
    X_i, X_j = np.meshgrid(x, x, indexing="ij")
    K = np.exp(-0.5 * ((X_i - X_j) / ell) ** 2)

    # Make sure it's Hermitian (numerically)
    K = 0.5 * (K + K.conj().T)

    return x, dx, psi, K

# ------------------------------------------------------------
# 3) Invariant J = <ψ, K ψ>
# ------------------------------------------------------------
def invariant_J(psi, K, dx=1.0):
    # vdot = conjugate dot product ψ^† (K ψ)
    return np.vdot(psi, K @ psi) * dx

# ------------------------------------------------------------
# 4) Information-augmented invariant
#     J_info = ψ^† K ψ - λ k_B T ∑ p log p
# ------------------------------------------------------------
def invariant_J_info(psi, K, dx=1.0, lam=1.0, k_B=1.0, T=1.0):
    J_energy = invariant_J(psi, K, dx)
    # probability density over lattice
    prob = np.abs(psi) ** 2
    prob /= prob.sum()                      # normalize to 1
    # avoid log(0)
    eps = 1e-30
    S_info = -np.sum(prob * np.log(prob + eps))
    J_total = J_energy - lam * k_B * T * S_info
    return J_total, J_energy, S_info

# ------------------------------------------------------------
# 5) Demo: show invariance under unitary change of basis
# ------------------------------------------------------------
def demo_quantum_foam():
    rng = np.random.default_rng(1234)

    # Build foam
    N = 128
    x, dx, psi, K = build_field_and_kernel(
        N=N, L=10.0, sigma=1.0, k_phase=3.0, ell=1.5
    )

    # Original invariants
    J0 = invariant_J(psi, K, dx)
    J_info0, J_energy0, S0 = invariant_J_info(psi, K, dx, lam=0.5, T=1.0)

    print("=== Original basis ===")
    print(f"J (energy invariant)           : {J0}")
    print(f"J_info (energy - λ k_B T S)    : {J_info0}")
    print(f"   energy part Re(J_energy)    : {J_energy0.real}")
    print(f"   entropy S_info              : {S0}")
    print()

    # Random unitary "change of foam coordinates"
    U = random_unitary(N, rng)
    psi_prime = U @ psi
    K_prime = U @ K @ U.conj().T

    J1 = invariant_J(psi_prime, K_prime, dx)
    J_info1, J_energy1, S1 = invariant_J_info(psi_prime, K_prime, dx, lam=0.5, T=1.0)

    print("=== Rotated basis (ψ' = U ψ, K' = U K U†) ===")
    print(f"J' (should equal J)            : {J1}")
    print(f"J_info'                        : {J_info1}")
    print(f"   energy part Re(J_energy')   : {J_energy1.real}")
    print(f"   entropy S_info'             : {S1}")
    print()
    print(f"|J' - J|                       : {abs(J1 - J0)}")
    print(f"|J_info' - J_info|             : {abs(J_info1 - J_info0)}")

    # --------------------------------------------------------
    # 6) Show that *non*-unitary change really změní pěnu
    #    (změníme kernel fyzicky, ne jen bázi)
    # --------------------------------------------------------
    noise_level = 0.05
    noise = noise_level * (
        rng.normal(size=K.shape) + 1j * rng.normal(size=K.shape)
    )
    noise = 0.5 * (noise + noise.conj().T)   # Hermitian noise
    K_pert = K + noise

    J2 = invariant_J(psi, K_pert, dx)
    J_info2, J_energy2, S2 = invariant_J_info(psi, K_pert, dx, lam=0.5, T=1.0)

    print("\n=== Physically changed kernel (K + noise) ===")
    print(f"J_pert                         : {J2}")
    print(f"J_info_pert                    : {J_info2}")
    print(f"|J_pert - J|                   : {abs(J2 - J0)}")
    print(f"|J_info_pert - J_info|         : {abs(J_info2 - J_info0)}")

    # --------------------------------------------------------
    # 7) Optional: visualization of |ψ|² a spektrum kernelu
    # --------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Probability density on lattice
    prob = np.abs(psi) ** 2
    prob /= prob.sum()
    axs[0].plot(x, prob)
    axs[0].set_title(r"Probability density $|\Psi(x)|^2$")
    axs[0].set_xlabel("x (lattice coordinate)")
    axs[0].set_ylabel("probability")

    # Eigenvalues of K
    evals = np.linalg.eigvalsh(K)
    axs[1].plot(evals.real, ".-")
    axs[1].set_title("Spectrum of kernel K (geometry / correlations)")
    axs[1].set_xlabel("mode index")
    axs[1].set_ylabel("eigenvalue")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_quantum_foam()
