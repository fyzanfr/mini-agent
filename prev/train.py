"""Training script — agent modifies hyperparameters."""
import pickle
import time
import random

# Load data
with open("data.pkl", "rb") as f:
    data = pickle.load(f)

# === AGENT CAN MODIFY BELOW ===
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 3
DEPTH = 2  # model "depth"
# === AGENT CAN MODIFY ABOVE ===

# Dummy training loop
random.seed(42)
best_loss = float('inf')

for epoch in range(EPOCHS):
    total = 0
    for i in range(0, len(data["train"]), BATCH_SIZE):
        batch = data["train"][i:i+BATCH_SIZE]
        # Simulate loss: lower LR + bigger batch + more depth = better
        loss = 1.0 / (LEARNING_RATE * 1000 + 1) + 10.0 / BATCH_SIZE + 5.0 / max(DEPTH, 1)
        loss += random.random() * 0.1  # noise
        total += loss
    
    avg = total / (len(data["train"]) // BATCH_SIZE)
    best_loss = min(best_loss, avg)
    print(f"Epoch {epoch+1}: loss={avg:.4f}")

# Metric: bits per byte (dummy — lower is better)
val_loss = best_loss + random.random() * 0.05
val_bpb = val_loss  # simplified

print(f"val_bpb: {val_bpb:.4f}")
