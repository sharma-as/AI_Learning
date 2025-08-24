import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(8,5))

# Example sentence tokens
tokens = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"]
x_positions = range(len(tokens))

# Plot tokens as text nodes
for i, token in enumerate(tokens):
    ax.text(x_positions[i], 1, token, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1))

# Draw attention arrows (illustrative, not exact weights)
attention_pairs = [(7,1), (7,0), (9,7)]  # "it"->"cat", "it"->"The", "tired"->"it"
for src, tgt in attention_pairs:
    ax.annotate("",
                xy=(x_positions[tgt], 1.05), xycoords='data',
                xytext=(x_positions[src], 1.25), textcoords='data',
                arrowprops=dict(arrowstyle="->", lw=1.5, color="red"))

# Styling
ax.set_xlim(-1, len(tokens))
ax.set_ylim(0.5, 2)
ax.axis("off")
ax.set_title("Self-Attention Example: 'The cat sat on the mat because it was tired.'", fontsize=12, pad=15)

plt.tight_layout()
plt.show()
