import matplotlib.pyplot as plt

def plot_confidence(nums_samples: list,
                    confidence: list):
    plt.figure(figsize=(8, 5))
    plt.plot(nums_samples, confidence, marker='o')
    plt.xscale('log')
    plt.xlabel('Number of samples (log scale)')
    plt.ylabel(r'$\gamma$')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()