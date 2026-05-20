"""Data preparation — run once."""
import pickle

# Dummy dataset: random text tokens
DATA = {
    "train": list(range(10000)),
    "val": list(range(1000)),
}

with open("data.pkl", "wb") as f:
    pickle.dump(DATA, f)

print("Data prepared.")
