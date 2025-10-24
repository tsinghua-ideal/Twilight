"""Some states and data structures."""

from enum import Enum

import torch
from torch import nn
import numpy as np

from typing import List


class CompressorType(Enum):
    NONE = 0
    SNAP_KV = 1

    @classmethod
    def from_str(cls, algo_str: str) -> "CompressorType":
        # print(algo_str)
        return cls[algo_str.upper()]


class IndexSelectorType(Enum):
    NONE = 0
    STREAMING = 1
    QUEST = 2
    ORACLE_TOPK = 3
    TIDAL_DECODE = 4
    SPARQ = 5
    DS = 6
    PQCACHE = 7

    @classmethod
    def from_str(cls, algo_str: str) -> "IndexSelectorType":
        # print(algo_str)
        return cls[algo_str.upper()]


class WeightEstimatorType(Enum):
    """The type of weight estimator of Twilight Optimizer."""

    NONE = 0
    MIN_MAX_QUANT = 1
    MAX_QUANT = 2
    THRESHOLD = 5
    H_THRESHOLD = 6

    @classmethod
    def from_str(cls, algo_str: str) -> "WeightEstimatorType":
        # print(algo_str)
        return cls[algo_str.upper()]


class WeightPrunerType(Enum):
    """The type of pruner applied to estimated weights."""

    NONE = 0
    THRESHOLD = 1
    TOP_P = 2

    @classmethod
    def from_str(cls, algo_str: str) -> "WeightPrunerType":
        # print(algo_str)
        return cls[algo_str.upper()]


class LocalState:
    """Store any information pass to the attention layer.

    In Python, `class` is an immutable type. We will set a global state which each LLM layer
    can access and update.
    """

    def __init__(self) -> None:
        self.arg_set = False

    def set_args(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.arg_set = True


class HistoryAccumulatedScoreInfo:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # A 3-dim array
        # Query 0 -> [Layer 0: [Head 0: accumulated_score]]
        # Layer as a dict
        # Head stored as a torch Tensor
        self._score_sum = []
        self._min_sum = []
        self._last_layer_id = -999

    def update(
        self, layer_id: int, attn_weights: torch.Tensor, mask: torch.Tensor
    ) -> None:
        norm_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        norm_weights[~mask] = 0.0
        score_sum = torch.sum(norm_weights, dim=-1).flatten()

        # print("Verify rate sum: ", layer_id, score_sum, torch.min(score_sum))

        if layer_id != self._last_layer_id + 1:  # New query
            self._score_sum.append({})
            self._min_sum.append({})

        self._score_sum[-1][layer_id] = score_sum
        self._min_sum[-1][layer_id] = torch.min(score_sum).item()
        self._last_layer_id = layer_id

    def print_min_sum_single_query(self, query_index: int = -1) -> None:
        query_score_info = self._min_sum[query_index]
        for layer_id, layer_score_info in query_score_info.items():
            print(f"Layer {layer_id}: {layer_score_info:.2f}")

    def print_score_sum_single_query(self, query_index: int = -1) -> None:
        query_score_info = self._score_sum[query_index]
        for layer_id, layer_score_info in query_score_info.items():
            print(f"Layer {layer_id}: {layer_score_info}")

    def get_avg_score_per_query(self) -> List[float]:
        ret = []
        for query_score_info in self._min_sum:
            query_avg = []
            for layer_id, layer_score_info in query_score_info.items():
                if layer_id > 1:
                    query_avg.append(layer_score_info)
            ret.append(np.mean(query_avg).tolist())
        return ret

    def get_total_avg_score(self) -> float:
        return float(np.mean(self.get_avg_score_per_query()))


class HistoryBudgetInfo:
    """Statistics about history budget information."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # A 3-dim array
        # Query 0 -> [Layer 0: [Head 0: budget_num]]
        # Layer as a dict
        # Head stored as a torch Tensor
        self._budget_num = []
        self._B0 = []
        self._last_layer_id = -999

    def update(self, layer_id: int, selected_mask: torch.Tensor) -> None:
        """Record current budget info."""

        head_budgets = torch.sum(selected_mask, dim=-1).flatten()
        if layer_id != self._last_layer_id + 1:  # New query
            self._budget_num.append({})

        self._budget_num[-1][layer_id] = head_budgets
        self._last_layer_id = layer_id

    def update_B0(self, layer_id: int, token_budget: int) -> None:
        """Record current budget info of B0."""

        if layer_id != self._last_layer_id + 1:  # New query
            self._B0.append({})

        self._B0[-1][layer_id] = token_budget

    def print_budget_info_single_query_with_B0(self, query_index: int = -1) -> None:
        query_budget_info = self._budget_num[query_index]
        total_budget = 0
        total_B0 = 0

        for layer_id, layer_budget_info in query_budget_info.items():
            cur_B0 = self._B0[query_index][layer_id]
            avg_budget = torch.mean(layer_budget_info.float()).item()
            if layer_id > 1:
                sum_budget = torch.sum(layer_budget_info).item()
                total_budget += sum_budget
                total_B0 += cur_B0
            print(
                f"Layer {layer_id}: {layer_budget_info.tolist()} B0: {cur_B0} Head Average: {avg_budget:.2f}"
            )

        print(f"Mean B0: {total_B0 / 30}")
        print(f"Mean B1: {total_budget / 30 / 32:.2f}", flush=True)

    def print_budget_info_single_query(self, query_index: int = -1) -> None:
        query_budget_info = self._budget_num[query_index]
        for layer_id, layer_budget_info in query_budget_info.items():
            sum_budget = torch.sum(layer_budget_info).item()
            avg_budget = torch.mean(layer_budget_info.float()).item()
            print(
                f"Layer {layer_id}: {layer_budget_info.tolist()} Head Average: {avg_budget:.2f}"
            )

    def get_avg_budget_per_layer_single_query(self, query_index: int) -> List[float]:
        query_budget_info = self._budget_num[query_index]
        ret = []
        for layer_id, layer_budget_info in query_budget_info.items():
            avg_budget = torch.mean(layer_budget_info.float()).item()
            ret.append(avg_budget)
        return ret

    def get_avg_budget_cur_query(self) -> float:
        query_budget_info = self._budget_num[-1]
        ret = []
        for layer_id, layer_budget_info in query_budget_info.items():
            avg_budget = torch.mean(layer_budget_info.float()).item()
            ret.append(avg_budget)
        return ret

    def get_avg_budget_per_query(self) -> List[float]:
        ret = []
        for query_budget_info in self._budget_num:
            query_avg = []
            for layer_id, layer_budget_info in query_budget_info.items():
                if layer_id > 1:
                    avg_budget = torch.mean(layer_budget_info.float()).item()
                    query_avg.append(avg_budget)
            ret.append(np.mean(query_avg).tolist())
        return ret

    def get_avg_budget_per_query_B0(self) -> List[float]:
        ret = []
        for query_B0 in self._B0:
            ret.append(float(np.mean(list(query_B0.values())[1:])))
        return ret

    def get_total_avg_budget(self) -> float:
        return float(np.mean(self.get_avg_budget_per_query()))

    def get_total_avg_budget_B0(self) -> float:
        return float(np.mean(self.get_avg_budget_per_query_B0()))
