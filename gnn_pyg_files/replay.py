from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List

from torch_geometric.data import HeteroData


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items: Deque[HeteroData] = deque(maxlen=capacity)

    def add(self, item: HeteroData) -> None:
        self.items.append(item)

    def extend(self, items: Iterable[HeteroData]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[HeteroData]:
        if batch_size >= len(self.items):
            return list(self.items)
        return self.rng.sample(list(self.items), batch_size)

    def __len__(self) -> int:
        return len(self.items)
