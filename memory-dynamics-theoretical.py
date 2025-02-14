import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 22})

# Model parameters 
N_v = 50
N_h = 4
T = 2000
tau_Xi = 250
noise_level = 0.4
beta = 2  #  beta 2 is already enough to see the effect

np.random.seed(41)

def create_orthogonal_memories(N_h, N_v):
    Xi = np.zeros((N_h, N_v))
    for i in range(N_h):
        Xi[i, :] = np.random.randn(N_v)
        for j in range(i):
            Xi[i, :] -= np.dot(Xi[i, :], Xi[j, :]) / np.dot(Xi[j, :], Xi[j, :]) * Xi[j, :]
        Xi[i, :] /= np.linalg.norm(Xi[i, :])
        Xi[i, :] *= np.sqrt(N_v)
    return Xi

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-4)

def softmax(x, beta):
    exp_x = np.exp(beta * (x - np.max(x)))
    return exp_x / np.sum(exp_x)

# Initialize memories and run simulation
Xi = create_orthogonal_memories(N_h, N_v)
Xi_new = Xi.copy()
for i in range(N_h):
    Xi_new[i, :] += noise_level * np.random.randn(N_v)

similarities = np.zeros((T, N_h))
Xi_current = Xi.copy()
Lambda = np.zeros((N_h, N_v))

# Run simulation
for t in range(T):
    v = Xi_new[t % N_h, :]
    S = Xi_current.dot(v)
    
    row_weights = softmax(S, beta)
    term = (-Xi_current + Lambda + v) * row_weights[:, np.newaxis]
    dXi = term / tau_Xi
    Xi_current += dXi
    
    for i in range(N_h):
        similarities[t, i] = cosine_similarity(Xi_current[i, :], Xi_new[i, :])

# Compute theoretical curves with correct normalization
def theoretical_similarity(t, tau, alpha):
    exp_term = np.exp(-t/tau)
    numerator = exp_term * alpha + (1 - exp_term)
    denominator =  np.sqrt(exp_term**2 + (1 - exp_term)**2 + 2*exp_term*(1-exp_term)*alpha)
    return numerator / denominator

# Estimate initial overlaps and time constants
initial_overlaps = np.array([cosine_similarity(Xi[i, :], Xi_new[i, :]) for i in range(N_h)])

p_mu = np.ones(N_h) / N_h
time_constants = tau_Xi / p_mu 

t_range = np.arange(T)
theoretical_curves = np.zeros((T, N_h))
for i in range(N_h):
    theoretical_curves[:, i] = theoretical_similarity(t_range, time_constants[i], initial_overlaps[i])


theoretical_alpha = 1 / np.sqrt(1 + noise_level**2)
print(f"Theoretical α: {theoretical_alpha}")
print(f"Computed α's: {initial_overlaps}")


# Create plot with both experimental and theoretical curves
plt.figure(figsize=(8, 6))
for i in range(N_h):
    plt.plot(t_range, similarities[:, i], label=f'Memory {i+1}')
    plt.plot(t_range, theoretical_curves[:, i], '--', color=plt.gca().lines[-1].get_color())

# plt.title(r'Memory Similarity Evolution')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Cosine Similarity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# save figure with high resolution and in pdf format
plt.savefig('memory-dynamics-theoretical.png', dpi=300)
plt.show()

# Print comparison metrics
print("Final experimental vs theoretical similarities:")
for i in range(N_h):
    print(f"Memory {i+1}: {similarities[-1, i]:.3f} vs {theoretical_curves[-1, i]:.3f}")
# %%
