# fedavg_pytorch.py
import copy
import random
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset

# ---------- Model ----------
class Net(nn.Module):
    def __init__(self, in_channels=1, n_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # compute flattened size for 32x32 input -> after conv/pool sizes match below
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # -> (N,6,28,28) assuming input 32x32
        x = F.max_pool2d(x, 2)     # -> (N,6,14,14)
        x = F.relu(self.conv2(x))  # -> (N,16,10,10)
        x = F.max_pool2d(x, 2)     # -> (N,16,5,5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------- Utilities ----------
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Partition dataset into K clients (IID split)
def iid_partition(dataset: Dataset, K: int) -> List[List[int]]:
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    shards = []
    size = n // K
    for k in range(K):
        start = k * size
        end = start + size if k < K-1 else n
        shards.append(idxs[start:end])
    return shards

# Simple non-IID partition by sorting labels into shards (like papers often do)
def noniid_partition(dataset: Dataset, K: int, shards_per_client: int = 2) -> List[List[int]]:
    # dataset is torchvision MNIST-like with .targets attribute
    # create 2*K shards, sort by label, assign shards to clients randomly
    labels = torch.tensor(dataset.targets) if hasattr(dataset, "targets") else torch.tensor([dataset[i][1] for i in range(len(dataset))])
    n = len(dataset)
    num_shards = K * shards_per_client
    shard_size = n // num_shards
    idxs = list(range(n))
    # pair indices with labels and sort
    idxs_labels = sorted(idxs, key=lambda i: int(labels[i]))
    shards = []
    for s in range(num_shards):
        start = s * shard_size
        end = start + shard_size if s < num_shards - 1 else n
        shards.append(idxs_labels[start:end])
    # assign shards to clients
    client_idxs = [[] for _ in range(K)]
    shard_ids = list(range(num_shards))
    random.shuffle(shard_ids)
    for i, sid in enumerate(shard_ids):
        client_idxs[i % K].extend(shards[sid])
    return client_idxs

# Create a DataLoader for a given list of indices
def make_dataloader(dataset: Dataset, indices: List[int], batch_size: int, shuffle=True):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

# Evaluate model on test set
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

# ---------- Client ----------
class Client:
    def __init__(self, client_id: int, train_dataset: Dataset, indices: List[int], device: torch.device,
                 batch_size: int = 32, lr: float = 0.01, local_epochs: int = 1):
        self.id = client_id
        self.device = device
        self.train_loader = make_dataloader(train_dataset, indices, batch_size=batch_size, shuffle=True)
        self.num_samples = len(indices)
        self.lr = lr
        self.local_epochs = local_epochs

    def local_update(self, global_model: nn.Module) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Receives a global model (nn.Module), copies it locally, trains locally for local_epochs,
        and returns the updated state_dict and number of samples for weighting.
        """
        # copy model to avoid in-place modifications of the server model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = local_model(x)
                loss = crit(out, y)
                loss.backward()
                optimizer.step()

        # return the updated parameters (move to cpu to ease aggregation)
        return {k: v.cpu().detach() for k, v in local_model.state_dict().items()}, self.num_samples

# ---------- Server (aggregator) ----------
class Server:
    def __init__(self, global_model: nn.Module, clients: List[Client], device: torch.device):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    @staticmethod
    def aggregate(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        """
        Weighted averaging of state_dicts.
        updates: list of (state_dict, num_samples)
        returns averaged state_dict
        """
        total_samples = sum(n for _, n in updates)
        if total_samples == 0:
            raise ValueError("No samples to aggregate.")

        avg_state = {}
        # initialize accumulator with zeros of same shape/dtype as first client's tensors
        first_state = updates[0][0]
        for k in first_state.keys():
            avg_state[k] = torch.zeros_like(first_state[k], dtype=first_state[k].dtype, device="cpu")

        # accumulate
        for state_dict, n in updates:
            weight = n / total_samples
            for k, param in state_dict.items():
                avg_state[k] += param * weight

        return avg_state

    def distribute_and_aggregate(self, client_ids: List[int]) -> None:
        """
        For a chosen subset of clients, send global model, get local updates,
        and aggregate them into the new global model.
        """
        selected_clients = [self.clients[i] for i in client_ids]
        updates = []
        for c in selected_clients:
            updated_state, n = c.local_update(self.global_model)
            updates.append((updated_state, n))

        avg_state = Server.aggregate(updates)
        # update server model (in place) with averaged parameters
        self.global_model.load_state_dict(avg_state)

# ---------- Main federated training routine ----------
def run_fedavg(
    *,
    K: int = 10,
    rounds: int = 20,
    C: float = 0.5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
    iid: bool = True,
    seed: int = 42
):
    device = get_device()
    print("Using device:", device)

    torch.manual_seed(seed)
    random.seed(seed)

    # data transforms: resize MNIST to 32x32 to match model expectation
    transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # partition
    if iid:
        client_indices = iid_partition(train_dataset, K)
    else:
        client_indices = noniid_partition(train_dataset, K, shards_per_client=2)

    # create clients
    clients = []
    for i in range(K):
        c = Client(client_id=i, train_dataset=train_dataset, indices=client_indices[i],
                   device=device, batch_size=batch_size, lr=lr, local_epochs=local_epochs)
        clients.append(c)

    # server and global model
    global_model = Net().to(device)
    server = Server(global_model=global_model, clients=clients, device=device)

    # global test loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    rounds_to_sample = max(1, int(C * K))
    print(f"Federated training: K={K}, rounds={rounds}, client fraction C={C} -> {rounds_to_sample} clients/round")
    print(f"local_epochs={local_epochs}, batch_size={batch_size}, lr={lr}, iid={iid}")

    # initial evaluation
    test_loss, test_acc = evaluate(server.global_model, test_loader, device)
    print(f"Round 0 | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    for r in range(1, rounds + 1):
        # sample clients for this round
        selected = random.sample(range(K), rounds_to_sample)
        server.distribute_and_aggregate(selected)

        # evaluate
        test_loss, test_acc = evaluate(server.global_model, test_loader, device)
        print(f"Round {r:3d} | Selected {selected} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return server.global_model

# ---------- Run an example ----------
if __name__ == "__main__":
    # Quick settings: small experiment you can run on CPU/GPU
    final_model = run_fedavg(
        K=10,
        rounds=20,
        C=0.5,          # fraction of clients per round
        local_epochs=1, # E
        batch_size=32,  # B
        lr=0.02,        # local learning rate
        iid=True,
        seed=123
    )
    print("Federated training completed.")