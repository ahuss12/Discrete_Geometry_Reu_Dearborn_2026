from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List

from torch_geometric.data import HeteroData


class ReplayBuffer:
    
    #Replay buffer with uniform random sampling.

    #It only stores individual HeteroData training examples and returns a
    #Python list of examples from sample(...). PyG's DataLoader, used in
    #train.py, is what turns that list into mini-batches.
    

    def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = random.Random(seed)
        self.items: Deque[HeteroData] = deque(maxlen=self.capacity)

    def add(self, item: HeteroData) -> None:
        #Add one training example.
        self.items.append(item)

    def extend(self, items: Iterable[HeteroData]) -> None:
        #Add many training examples.
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[HeteroData]:
        
               #        Return a Python list of individual HeteroData examples.

        batch_size = int(batch_size)

        if batch_size <= 0 or len(self.items) == 0:
            return []

        if batch_size >= len(self.items):
            return list(self.items)

        # Convert to a list so random.sample can index it.
        return self.rng.sample(list(self.items), batch_size)

    def __len__(self) -> int:
        return len(self.items)



# future priority queue version(maybe?)
#     Uniform replay is easier to reason about while we are still changing the
#     graph representation and model. Priority replay can be useful later
#
# import heapq
# import itertools
# from typing import Tuple
#
# class PrioritizedReplayBuffer:

#
#     def __init__(self, capacity: int = 50_000, seed: int = 0) -> None:
#         self.capacity = int(capacity)
#         self.rng = random.Random(seed)
#         self.heap: list[Tuple[float, int, HeteroData]] = []
#         self.counter = itertools.count()
#
#     def add(self, item: HeteroData, priority: float = 1.0) -> None:
#         # Negative priority because heapq pops the smallest value first.
#         entry = (-float(priority), next(self.counter), item)
#         heapq.heappush(self.heap, entry)
#
#         #remove the lowest-priority item.
#         if len(self.heap) > self.capacity:
#             self.heap.sort()
#             self.heap = self.heap[: self.capacity]
#             heapq.heapify(self.heap)
#
#     def extend(self, items: Iterable[HeteroData], priority: float = 1.0) -> None:
#         for item in items:
#             self.add(item, priority=priority)
#
#     def sample(self, batch_size: int) -> List[HeteroData]:
#         #pop highest-priority examples.
#         batch = []
#         for _ in range(min(int(batch_size), len(self.heap))):
#             _, _, item = heapq.heappop(self.heap)
#             batch.append(item)
#         return batch
#
#     def __len__(self) -> int:
#         return len(self.heap)
