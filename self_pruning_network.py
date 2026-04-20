"""
Self-Pruning Neural Network on CIFAR-10
Tredence Analytics – AI Engineering Intern Case Study

This script implements a feed-forward neural network with learnable "gate" parameters
that allow the network to prune its own weights during training via L1 sparsity loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# ─────────────────────────────────────────────
# PART 1 – PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that associates each weight with a
    learnable scalar gate in [0, 1].  During the forward pass the weight is
    element-wise multiplied by sigmoid(gate_scores), so a gate that collapses
    to 0 effectively removes the corresponding weight from the network.

    Gradients flow correctly through both `weight` and `gate_scores` because
    all operations (sigmoid, element-wise multiply, matrix multiply) are
    differentiable in PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight & bias – identical initialisation to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None

        # Gate scores – same shape as weight, registered as a parameter so
        # the optimiser updates them alongside the weights.
        # Initialised near 0 so sigmoid gives values close to 0.5 (neutral).
        self.gate_scores = nn.Parameter(torch.zeros_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squash gate_scores into (0, 1) with sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Element-wise multiply: gates near 0 suppress the weight entirely
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard linear operation – F.linear handles the matmul + bias
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return gate values (after sigmoid) as a detached tensor."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (i.e. effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────
# Network definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A simple 3-hidden-layer feed-forward network for CIFAR-10.
    CIFAR-10 images are 32×32×3 = 3072-dimensional when flattened.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)     # 10 CIFAR-10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)                     # raw logits for CrossEntropyLoss

    def prunable_layers(self):
        """Iterate over all PrunableLinear layers."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values across every PrunableLinear layer.
        This is always non-negative, so minimising it pushes gates toward 0.
        """
        total = torch.tensor(0.0, requires_grad=True)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of all gates that are effectively pruned."""
        pruned = total = 0
        for layer in self.prunable_layers():
            g = layer.get_gates()
            pruned += (g < threshold).sum().item()
            total  += g.numel()
        return pruned / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value into a single numpy array."""
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(vals)


# ─────────────────────────────────────────────
# PART 2 / 3 – Training & Evaluation
# ─────────────────────────────────────────────

def get_loaders(batch_size: int = 256):
    """Return CIFAR-10 train and test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def train(lam: float,
          epochs: int = 20,
          lr: float = 1e-3,
          device: str = "cpu") -> dict:
    """
    Train a SelfPruningNet with sparsity coefficient `lam`.

    Returns a dict with keys:
        test_accuracy, sparsity_level, gate_values, history
    """
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lam}  |  epochs={epochs}  |  device={device}")
    print(f"{'='*55}")

    train_loader, test_loader = get_loaders()
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "test_acc": [], "sparsity": []}

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            # Total Loss = CrossEntropy + λ * L1(gates)
            cls_loss = F.cross_entropy(logits, labels)
            sp_loss  = model.sparsity_loss()
            loss     = cls_loss + lam * sp_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # ── evaluation ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        acc      = 100.0 * correct / total
        sparsity = 100.0 * model.overall_sparsity()
        avg_loss = running_loss / len(train_loader)

        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)
        history["sparsity"].append(sparsity)

        print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
              f"acc={acc:.2f}% | sparsity={sparsity:.1f}% | "
              f"time={time.time()-t0:.1f}s")

    final_acc      = history["test_acc"][-1]
    final_sparsity = history["sparsity"][-1]

    print(f"\n  ✓ Final Test Accuracy : {final_acc:.2f}%")
    print(f"  ✓ Final Sparsity      : {final_sparsity:.1f}%")

    return {
        "test_accuracy":  final_acc,
        "sparsity_level": final_sparsity,
        "gate_values":    model.all_gate_values(),
        "history":        history,
        "model":          model,
    }


def plot_gate_distribution(gate_values: np.ndarray,
                           lam: float,
                           save_path: str = "gate_distribution.png"):
    """
    Plot the distribution of final gate values.
    A successful run shows a large spike at 0 and a cluster away from 0.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gate_values, bins=100, color="#4C72B0", edgecolor="white",
            linewidth=0.3)
    ax.set_xlabel("Gate value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=13)
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Gate distribution plot saved → {save_path}")


def plot_comparison(results: dict, save_path: str = "lambda_comparison.png"):
    """Plot accuracy and sparsity vs lambda."""
    lambdas   = list(results.keys())
    accs      = [results[l]["test_accuracy"]  for l in lambdas]
    sparsities= [results[l]["sparsity_level"] for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar([str(l) for l in lambdas], accs, color=["#4C72B0","#DD8452","#55A868"])
    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Test Accuracy vs λ", fontsize=13)
    ax1.set_ylim(0, 100)
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    ax2.bar([str(l) for l in lambdas], sparsities, color=["#4C72B0","#DD8452","#55A868"])
    ax2.set_xlabel("Lambda (λ)", fontsize=12)
    ax2.set_ylabel("Sparsity Level (%)", fontsize=12)
    ax2.set_title("Sparsity Level vs λ", fontsize=13)
    ax2.set_ylim(0, 100)
    for i, v in enumerate(sparsities):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Lambda comparison plot saved → {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Three lambda values: low / medium / high
    lambdas = [1e-5, 1e-4, 1e-3]
    EPOCHS  = 20       # increase to 40+ for better accuracy

    all_results = {}
    for lam in lambdas:
        all_results[lam] = train(lam, epochs=EPOCHS, device=device)

    # ── Print summary table ──
    print("\n\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print("  " + "-"*40)
    for lam in lambdas:
        r = all_results[lam]
        print(f"  {lam:<12} {r['test_accuracy']:>14.2f} {r['sparsity_level']:>14.1f}")
    print("="*55)

    # ── Plots ──
    # Gate distribution for the medium-lambda model (best balance)
    best_lam = lambdas[1]
    plot_gate_distribution(all_results[best_lam]["gate_values"],
                           lam=best_lam,
                           save_path="gate_distribution.png")

    plot_comparison(all_results, save_path="lambda_comparison.png")

    print("\nAll done. Check gate_distribution.png and lambda_comparison.png.")