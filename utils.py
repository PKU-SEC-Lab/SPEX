from sglang.lang.ir import SglSamplingParams
from sglang.global_config import global_config
import asyncio
import aiohttp
from typing import Optional, List, Union
import time
import json
from collections import defaultdict
from math_evaluate import extract_shepherd_answer, grade_answer
import torch
import math
import copy
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import threading
lock = threading.Lock()

# Process-wide wall-clock origin so timings from multiple threads share an axis.
_EPOCH = time.time()


def now_ts():
    """Seconds since process start (shared epoch across all trees/threads)."""
    return time.time() - _EPOCH


class BudgetCoordinator:
    """Cross-tree (inter-query) budget coordinator for speculation.

    Tracks how many policy generations are currently in flight across all
    threads.  Each generation is tagged either 'main' (regular reward-guided
    expansion) or 'spec' (speculative expansion).  A generation is *claimed*
    at enqueue time and *released* when the producer task finishes the call.

    The available speculation budget at any moment is::

        avail_spec = max(0, total_budget - main_active - spec_active)

    which directly implements the user spec:
        spec_count = total_query_count - base_running_threads_count

    main_active counts in-flight main generations across all trees ("base
    running threads"); spec_active prevents over-claim when multiple trees
    enqueue spec concurrently.
    """

    def __init__(self, total_budget: int):
        self.total = int(total_budget)
        self.main_active = 0
        self.spec_active = 0
        # Stats
        self.peak_total = 0
        self.peak_main = 0
        self.peak_spec = 0
        self.spec_claims = 0
        self.spec_rejects = 0
        # Global per-request timing log (filled by Tree.producer).
        # Each entry: dict(query_id, parent_id, child_id, source,
        #   t_enqueue, t_pickup, t_done, depth, ok)
        self.request_log = []
        self._lock = threading.Lock()

    def claim(self, source: str):
        with self._lock:
            if source == "spec":
                self.spec_active += 1
                self.spec_claims += 1
            else:
                self.main_active += 1
            tot = self.main_active + self.spec_active
            if tot > self.peak_total:
                self.peak_total = tot
            if self.main_active > self.peak_main:
                self.peak_main = self.main_active
            if self.spec_active > self.peak_spec:
                self.peak_spec = self.spec_active

    def release(self, source: str):
        with self._lock:
            if source == "spec":
                if self.spec_active > 0:
                    self.spec_active -= 1
            else:
                if self.main_active > 0:
                    self.main_active -= 1

    def avail_spec(self) -> int:
        with self._lock:
            return max(0, self.total - self.main_active - self.spec_active)

    def try_claim_spec(self, n: int) -> int:
        """Atomically claim up to n spec slots; return slots actually claimed."""
        if n <= 0:
            return 0
        with self._lock:
            avail = max(0, self.total - self.main_active - self.spec_active)
            grant = min(n, avail)
            self.spec_active += grant
            self.spec_claims += grant
            self.spec_rejects += max(0, n - grant)
            tot = self.main_active + self.spec_active
            if tot > self.peak_total:
                self.peak_total = tot
            if self.spec_active > self.peak_spec:
                self.peak_spec = self.spec_active
            return grant

    def stats(self):
        with self._lock:
            return dict(
                total=self.total,
                main_active=self.main_active,
                spec_active=self.spec_active,
                peak_total=self.peak_total,
                peak_main=self.peak_main,
                peak_spec=self.peak_spec,
                spec_claims=self.spec_claims,
                spec_rejects=self.spec_rejects,
            )

    def record_request(self, entry: dict):
        with self._lock:
            self.request_log.append(entry)


# =============================================================================
# Shared inter-query helpers (used by both bfs_async.py and dfs_async.py).
# Centralising these keeps DFS and BFS on the same machinery:
# the same TOTAL_PARALLEL_BUDGET env var, the same budget-stats line, the same
# global_requests.json schema.  Tree subclasses still own per-query trace dump
# (their node fields differ) but they go through Tree.dump_trace.
# =============================================================================

def setup_inter_query_budget(paras, args, algo: str):
    """Configure paras["budget_coordinator"] from TOTAL_PARALLEL_BUDGET env.

    `algo` is "bfs" or "dfs" — only used for the informational log line so
    operators see the relevant capacity hint for the algorithm.
    Returns the coordinator (or None if env-var unset).
    """
    import os as _os
    if "TOTAL_PARALLEL_BUDGET" not in _os.environ:
        paras["budget_coordinator"] = None
        return None
    total_budget = int(_os.environ["TOTAL_PARALLEL_BUDGET"])
    coord = BudgetCoordinator(total_budget)
    paras["budget_coordinator"] = coord
    paras["total_parallel_budget"] = total_budget
    if algo == "bfs":
        capacity = args.num_threads * args.width
        print(f"[inter-query] TOTAL_PARALLEL_BUDGET={total_budget} (nt={args.num_threads}, width={args.width}; base capacity ~ nt*width = {capacity})")
    elif algo == "dfs":
        capacity = args.num_threads * args.wid * (args.num_spec + 1)
        print(f"[inter-query] TOTAL_PARALLEL_BUDGET={total_budget} (nt={args.num_threads}, wid={args.wid}, num_spec={args.num_spec}; nominal capacity ~ {capacity})")
    else:
        print(f"[inter-query] TOTAL_PARALLEL_BUDGET={total_budget}")
    return coord


def print_budget_stats(coord):
    """Print one-line [budget-stats] summary; return the dict for log_data, or None."""
    if coord is None:
        return None
    s = coord.stats()
    print(f"[budget-stats] total={s['total']} peak_total={s['peak_total']} peak_main={s['peak_main']} peak_spec={s['peak_spec']} spec_claims={s['spec_claims']} spec_rejects={s['spec_rejects']}")
    return s


def dump_global_request_log(coord, trace_dir, *, num_threads, width, speculative, algo, extra=None):
    """Dump cross-tree per-request timeline to <trace_dir>/global_requests.json.

    `width` is the per-tree fan-out (BFSTree.width or DFSTree.mcts_wid) — kept
    in the trace so analyzers can normalise.
    """
    if coord is None or not trace_dir:
        return
    import os as _os
    try:
        _os.makedirs(trace_dir, exist_ok=True)
        gpath = _os.path.join(trace_dir, "global_requests.json")
        s = coord.stats()
        body = {
            "epoch_offset": 0.0,
            "total_budget": s['total'],
            "num_threads": num_threads,
            "width": width,
            "speculative": speculative,
            "algo": algo,
            "requests": coord.request_log,
        }
        if extra:
            body.update(extra)
        with open(gpath, "w") as f:
            json.dump(body, f)
        print(f"[trace] global request log dumped: {gpath} ({len(coord.request_log)} entries)")
    except Exception as _e:
        print(f"[trace dump error] {_e}")


class State:
    def __init__(self, text="", policy_url="", reward_url="", scores=None, meta_info=None):
        self.text_ = text
        self.scores_ = scores if scores is not None else []
        self.meta_info = meta_info if meta_info is not None else {}
        self.policy_url = policy_url
        self.reward_url = reward_url    

    def text(self):
        return self.text_

    def scores(self):
        return self.scores_

    def get_meta_info(self, key):
        return self.meta_info.get(key, None)

    def set_meta_info(self, key, value):
        self.meta_info[key] = value

    def fork(self, number=1):
        return [copy.deepcopy(self) for _ in range(number)]


class TreeNode:
    def __init__(self, id, state, score, curr_text, num_step_tokens=0, parent=None):
        self.id = id 
        self.state = state
        self.text_ = state.text()
        self.curr_text = curr_text
        self.score_ = score
        self.parent = parent
        self.children = []
        self.leaf_ = False
        self.cum_tokens = 0
        self.num_step_tokens = num_step_tokens
        self.finish_time = 0
        if parent is not None and "The answer is" in self.text_:
            self.leaf_ = True
        if parent is not None:
            self.depth = parent.get_depth() + 1
            self.cum_tokens += num_step_tokens
        else:
            self.depth = 0
            self.cum_tokens = num_step_tokens
        self.answer = None
        # Trace metadata: how this node was generated and whether it ended up on the main path.
        self.created_via = "main"  # "main" | "spec" — set by producer based on request tag
        self.was_main_chosen = False  # set True if select_and_expand ever picks this node for further expansion
     
    def get_id(self):
        return self.id
    
    def get_parent(self):
        return self.parent
    
    def get_text(self):
        return self.text_

    def get_state(self):
        return self.state
    
    def get_depth(self):
        return self.depth
    
    def get_score(self):
        return self.score_
    
    def is_leaf(self):
        return self.leaf_
    
    def get_cum_tokens(self):
        return self.cum_tokens
    
class Tree:
    correct_count = 0
    total_count = 0
    conf_list=[]
    conf_list_right=[]
    conf_list_other=[]
    conf_list_other_right=[]
    saving_rate=[]
    def __init__(self, root_state, paras, reward_backend, Nodetype, num_producers=64):
        self.size_ = 1
        self.nodes = []
        self.paras = paras
        self.reward_backend = reward_backend
        self.remaining_width = paras["width"]
        self.history_list = []
        self.running_nodes = defaultdict(int)
        self.depth_nodes = [[] for i in range(100)]
        self.Nodetype = Nodetype
        self.root_ = self.Nodetype(0,root_state, 1.0,root_state.text())
        self.nodes.append(self.root_)
        self.depth_nodes[0].append(self.root_)

        self.request_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.num_producers = num_producers
        self.shutdown_event = asyncio.Event()
        self.producers = [
            asyncio.create_task(self.producer()) for _ in range(self.num_producers)
        ]
        self.running_producers = 0
        self.start_time = time.time()
        self.answer_time = []

        self.leaf_nodes = []
        self.spec_leaf_nodes = []
        self.lambdac=2
        self.lambdas=0
        # Optional shared inter-query budget coordinator (set via paras["budget_coordinator"]).
        self.budget_coordinator = paras.get("budget_coordinator", None)
        # Per-tree request log (also mirrored into coordinator if any).
        self.request_log = []
        # Will be filled in by _reward_guided_search.
        self.query_id = None
        
    
    def reset_running_list(self):
        self.running_list = []

    # ------------------------------------------------------------------
    # Per-query trace dump.  Subclasses override _trace_meta and/or
    # _trace_node_dict to add algorithm-specific fields; the JSON envelope
    # and file IO live here so DFS and BFS share one code path.
    # ------------------------------------------------------------------
    def _trace_node_dict(self, node):
        """Common per-node fields recorded in the trace."""
        return {
            "id": node.id,
            "depth": node.get_depth(),
            "score": float(node.get_score()) if node.get_score() is not None else None,
            "parent_id": node.parent.id if node.parent is not None else None,
            "finish_time": float(getattr(node, "finish_time", 0.0)),
            "is_leaf": bool(node.is_leaf()),
            "created_via": getattr(node, "created_via", "main"),
            "num_step_tokens": int(node.num_step_tokens),
        }

    def _trace_meta(self):
        """Common per-tree metadata recorded in the trace."""
        return {
            "start_time": self.start_time,
            "speculative": bool(self.paras.get("speculative", False)),
            "max_tokens": int(self.paras.get("max_tokens", 0)),
            "request_log": self.request_log,
        }

    def dump_trace(self, path):
        """Write the per-query trace JSON.  Called from _dfs / _reward_guided_search."""
        meta = self._trace_meta()
        meta["nodes"] = [self._trace_node_dict(n) for n in self.nodes]
        with open(path, "w") as f:
            json.dump(meta, f)
    
    def get_running_list(self):
        return self.running_list
    
    def get_history_list(self):
        return self.history_list
    
    def get_nodes(self):
        return self.nodes
    
    def insert(self, state, parent):
        if state.scores() == [] or state.scores == None:
            return
        score = state.scores()[-1]
        num_step_tokens = state.get_meta_info("step")["completion_tokens"]
        if parent is None:
            curr_text = state.text()
        else:
            parent_text_len = len(parent.get_text())
            curr_text = state.text()[parent_text_len:]
        new_node = self.Nodetype(self.size_, state, score, curr_text, num_step_tokens, parent)
        new_node.finish_time = time.time()-self.start_time
        if parent is not None:
            parent.children.append(new_node)
        self.size_ += 1
        depth = new_node.get_depth()
        self.depth_nodes[depth].append(new_node)
        self.nodes.append(new_node)
        if new_node.is_leaf() == True:
            answer = extract_shepherd_answer(new_node.get_text())
            new_node.answer = answer
            self.leaf_nodes.append(new_node)
            self.spec_leaf_nodes.append(new_node)
        return new_node
    
    async def expand_one_branch(self, node):
        fork = node.get_state().fork(1)[0]
        if self.paras["policy_model_type"] == "llemma":
            prompt = fork.text()
        else:
            prompt = f"""
            ONLY output one of these:
            1. "The answer is:[X]"
            2. "Think one step:[Y]"

            NEVER explain rules.
            NEVER add other text.

            Task: {fork.text()}
            """
        comp, meta_info = await generate_async(
            prompt,
            fork.policy_url,
            max_step_tokens= self.paras["max_step_tokens"],
            stop="Step " + str(node.get_depth() + 2),
            temperature=self.paras["temperature"],
        )
        if self.paras["policy_model_type"] != "llemma":
            comp = "Step " + str(node.get_depth() + 1) + ": " + comp + " ки\n"
        fork.text_ += comp
        fork.meta_info["step"] = meta_info
        comp, meta_info = await generate_async(
            fork.text(),
            fork.reward_url,
            max_step_tokens=0,
            forward_only=True,
            logits_require_id=8094
        )
        fork.scores_ = comp
        new_node = self.insert(fork, node)
        return new_node

    async def producer(self):
        while not self.shutdown_event.is_set():
            try:
                item = await asyncio.wait_for(self.request_queue.get(), timeout=5)
                # Items may be (node, source, t_enqueue), (node, source), or bare node.
                if isinstance(item, tuple):
                    if len(item) == 3:
                        node, source, t_enqueue = item
                    elif len(item) == 2:
                        node, source = item
                        t_enqueue = now_ts()
                    else:
                        node, source, t_enqueue = item[0], "main", now_ts()
                else:
                    node, source, t_enqueue = item, "main", now_ts()
                self.running_producers += 1
                t_pickup = now_ts()
                # Note: budget claim is done at enqueue time (in select_and_expand /
                # speculative_expand) so the cap actually limits how much we put
                # into the queue.  Use try/finally so the claim is released even
                # if the task is cancelled mid-generation during shutdown.
                released = False
                new_node = None
                try:
                    new_node = await self.expand_one_branch(node)
                    if new_node is not None:
                        new_node.created_via = source
                except asyncio.CancelledError:
                    if self.budget_coordinator is not None:
                        self.budget_coordinator.release(source)
                        released = True
                    self.running_producers -= 1
                    raise
                except Exception as e:
                    print(f"[producer error] node id: {getattr(node, 'id', None)}, error: {e}")
                    new_node = None
                t_done = now_ts()
                if not released and self.budget_coordinator is not None:
                    self.budget_coordinator.release(source)
                # Per-request timing record.
                entry = {
                    "query_id": self.query_id,
                    "parent_id": getattr(node, "id", -1),
                    "child_id": getattr(new_node, "id", -1) if new_node is not None else -1,
                    "depth": getattr(node, "depth", -1) + 1 if new_node is not None else getattr(node, "depth", -1),
                    "source": source,
                    "t_enqueue": t_enqueue,
                    "t_pickup": t_pickup,
                    "t_done": t_done,
                    "ok": new_node is not None,
                }
                self.request_log.append(entry)
                if self.budget_coordinator is not None:
                    self.budget_coordinator.record_request(entry)
                await self.response_queue.put((node, new_node))
                self.running_producers -= 1
            except asyncio.TimeoutError:
                continue

    async def shutdown(self):
        """Gracefully cancels and shuts down all producer tasks."""
        self.shutdown_event.set()
        # Cancel all running producer tasks
        for task in self.producers:
            task.cancel()
        
        # Wait for all tasks to acknowledge cancellation
        await asyncio.gather(*self.producers, return_exceptions=True)
        # Release any budget claims for queued-but-not-yet-processed items
        # so the global coordinator counters return to zero between queries.
        if self.budget_coordinator is not None:
            try:
                while True:
                    item = self.request_queue.get_nowait()
                    if isinstance(item, tuple):
                        src = item[1] if len(item) >= 2 else "main"
                    else:
                        src = "main"
                    self.budget_coordinator.release(src)
            except asyncio.QueueEmpty:
                pass
    
    def select_softmax(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num
    
    def select_softmax_with_truncation(self, node_list, node_weights, width):
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()
        nodes = []
        weights = []
        truncate_ratio = self.paras["truncate_ratio"]
        keep_num = int(math.ceil(len(sorted_node_weight_pair_list) * truncate_ratio))
        sorted_node_weight_pair_list = sorted_node_weight_pair_list[:keep_num]
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            weights.append(pair[1])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        for weight in exp_weights:
            num = int(math.ceil(width * weight / sum_exp_weights))
            select_num.append(num)
            width -= num
            sum_exp_weights -= weight
        return nodes, select_num


    def select_softmax_costmodel(self, node_list, node_weights, width, depth):

        # compute leaf mapping
        def compute_branch_leaf_mapping(leaf_list):
            branch_leaf_mapping = {}

            for leaf in leaf_list:
                current_node = leaf
                while current_node.parent is not None:  # Traverse up to the root
                    parent = current_node.parent
                    if parent.id not in branch_leaf_mapping:
                        branch_leaf_mapping[parent.id] = []
                    branch_leaf_mapping[parent.id].append(leaf.id)
                    current_node = parent  # Move up to the parent

            # Ensure leaf IDs are unique in each branch's list
            for branch_id in branch_leaf_mapping:
                branch_leaf_mapping[branch_id] = list(set(branch_leaf_mapping[branch_id]))

            return branch_leaf_mapping

        # compute mapping
        branch_leaf_mapping = compute_branch_leaf_mapping(node_list)

        # Decision variables
        N = len(node_list) # num leafs
        M = len(branch_leaf_mapping) # num branches
        x = [LpVariable(f"x_{i}", cat="Binary") for i in range(N)]  # Binary decision for each leaf
        y = [LpVariable(f"y_{j}", cat="Binary") for j in range(M)]  # Binary decision for each branch

        # Objective function: maximize outcome scores plus costs
        if self.lambdac == 0: # guard
            lambdac = 1e-4
        else:
            lambdac = -self.lambdac

        # build a map here of leaf node id -> index of binary variable
        leafnodemap = {}
        for i in range(N):
            leafnodemap[node_list[i].id] = i

        # build a map here of branch node id -> index of binary variable
        branchnodemap = {}
        idx = 0
        for j, leaf_nodes in branch_leaf_mapping.items():
            branchnodemap[j] = idx
            idx += 1

        # Define the problem
        problem = LpProblem("Tree_Selection_Problem", LpMaximize)
        Cost = [1/(M+N) for i in range(M+N)] # for now, set cost as 1/num_branch_nodes for each, later set it based on num_tokens

        # use num_retained_trajectories as normalized outcome score
        node_weight_pair_list = [(i, node, weight) for i, (node, weight) in enumerate(zip(node_list, node_weights))]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda triplet: triplet[2], reverse=True)
        weights = []
        for triplet in sorted_node_weight_pair_list:
            weights.append(triplet[2])
        weights = torch.tensor(weights)
        T = self.paras["softmax_temperature"]
        exp_weights = torch.exp(weights / T)
        sum_exp_weights = exp_weights.sum()
        outcome_score = []
        width_tmp = width
        for weight in exp_weights:
            if sum_exp_weights > 0:
                num = int(math.ceil(width_tmp * weight / sum_exp_weights))
                outcome_score.append(num)
                width_tmp -= num
                sum_exp_weights -= weight
            else:
                outcome_score.append(0)

        score_in_original_order = [0] * len(node_weights)
        for i, (orig_idx, node, weight) in enumerate(sorted_node_weight_pair_list):
            score_in_original_order[orig_idx] = outcome_score[i]
        O = [score_in_original_order[i] / sum(score_in_original_order) for i in range(len(score_in_original_order))]

        # Collect sentences from node_list
        sentences = []
        for i in range(N):
            if node_list[i].parent is not None:
                prevlen = len(node_list[i].parent.get_text())
            else:
                prevlen = 0
            s = node_list[i].get_text()
            s = s[prevlen:]
            sentences.append(s)

        # Compute the number of unique sequences
        unique_sequences = set(sentences)
        num_unique = len(unique_sequences)

        # if similarity
        lambdas = self.lambdas
        apply_sim = lambdas > 0 and N > 1 and num_unique > 1
        if apply_sim:
            # produce sequence embeddings here
            embeddings = self.model.encode(sentences, batch_size=64)
            embeddings = np.array(embeddings)

            # clustering sequences
            Z = linkage(embeddings, method='average', metric='cosine')
            clusters = fcluster(Z, 0.05, criterion='distance') - 1 # subtract to get 0-indexed labels
            K = len(np.unique(clusters))

            # link coverage variables to cluster selections
            coverage = [LpVariable(f"coverage_{i}", cat="Binary") for i in range(K)]  # Binary decision for each cluster
            for k in range(K):
                cluster_indices = [i for i, c in enumerate(clusters) if c == k]
                problem += lpSum(x[i] for i in cluster_indices) >= coverage[k], f"Coverage_Lower_{k}"

            # divide lambdas by num_clusters to scale term appropriately
            lambdas /= K

            # objective function
            problem += (
                lpSum(O[i] * x[i] for i in range(N)) +  # Maximize outcomes for selected leaves
                lpSum(lambdac * Cost[j] * y[j] for j in range(M)) +  # Include costs for branch nodes
                lpSum(lambdac * Cost[j+M] * x[j] for j in range(N)) + # Include costs for leaf nodes
                lpSum(lambdas * coverage[k] for k in range(K)) # include similarity enforcement
            ), "Objective"

        else:

            # objective function
            problem += (
                lpSum(O[i] * x[i] for i in range(N)) +  # Maximize outcomes for selected leaves
                lpSum(lambdac * Cost[j] * y[j] for j in range(M)) +  # Include costs for branch nodes
                lpSum(lambdac * Cost[j+M] * x[j] for j in range(N)) # Include costs for leaf nodes
            ), "Objective"

        # Constraints - Link y (branch selected) to x (leaf nodes in the branch)
        for j, leaf_nodes in branch_leaf_mapping.items():
            branch_node_idx = branchnodemap[j]
            leaf_nodes_mapped = [leafnodemap[i] for i in leaf_nodes]

            problem += y[branch_node_idx] <= lpSum(x[i] for i in leaf_nodes_mapped), f"Branch_{j}_activation"
            for i in leaf_nodes_mapped:
                problem += y[branch_node_idx] >= x[i], f"Branch_{j}_leaf_{i}_link"

        # constraint to not prune all nodes
        problem += lpSum(x[i] for i in range(N)) >= 1, "Keep_Node_Constraint"

        # Solve the problem
        problem.solve(PULP_CBC_CMD(msg=0))

        # decide which nodes to keep
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, node_weights)]
        weights = []
        idx = 0
        T = self.paras["softmax_temperature"]
        for pair in node_weight_pair_list:
            keep_node = x[idx].value() == 1
            if keep_node:
                weights.append(pair[1] / T)
            else:
                weights.append(float('-inf'))
            idx += 1

        # re-apply rebase-style sampling
        node_weight_pair_list = [(node, weight) for node, weight in zip(node_list, weights)]
        sorted_node_weight_pair_list = sorted(node_weight_pair_list, key=lambda pair: pair[1])
        sorted_node_weight_pair_list.reverse()

        nodes = []
        sorted_weights = []
        for pair in sorted_node_weight_pair_list:
            nodes.append(pair[0])
            sorted_weights.append(pair[1])

        # compute rebase weighting
        weights = torch.tensor(sorted_weights)
        exp_weights = torch.exp(weights)
        sum_exp_weights = exp_weights.sum()
        select_num = []
        nodes_kept = []
        idx = 0
        for weight in exp_weights:
            if sum_exp_weights > 0:
                num = int(math.ceil(width * weight / sum_exp_weights))
                if num > 0:
                    nodes_kept.append(nodes[idx])
                select_num.append(num)
                width -= num
                sum_exp_weights -= weight
            else:
                select_num.append(0)
            idx += 1



        return nodes, select_num
    
default_sampling_para = SglSamplingParams(
    max_new_tokens=256,
    stop=(),
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    ignore_eos=False,
    forward_only=False,
    logits_require_id=None,
)

def _resolve_sampling_params(sampling_params):
    clone = None 
    for item in [
        "max_new_tokens",
        "stop",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "ignore_eos",
        "dtype",
        "regex",
        "forward_only",
        "logits_require_id",
    ]:
        value = getattr(sampling_params, item, None)
        if value is not None:
            if clone is None:
                clone = default_sampling_para.clone() 
            setattr(clone, item, value)

    return clone or default_sampling_para 


async def generate_async(
    text: str,
    base_url: str,
    max_step_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    forward_only: bool = False,
    logits_require_id: Optional[int] = None,
):
    raw_sampling_params = SglSamplingParams(
        max_new_tokens=max_step_tokens,
        stop=stop,
        temperature=temperature,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        ignore_eos=None,
        dtype=None,
        regex=None,
        forward_only=forward_only,
        logits_require_id=logits_require_id,
    )
    sampling_params = _resolve_sampling_params(raw_sampling_params)
    if sampling_params.dtype  is None:
        data = {
            "text": text,
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output, 
                **sampling_params.to_srt_kwargs(), 
            },
        }
    elif sampling_params.dtype  in [int, "int"]:
        data = {
            "text": text, 
            "sampling_params": {
                "skip_special_tokens": global_config.skip_special_tokens_in_output, 
                "dtype": "int",
                **sampling_params.to_srt_kwargs(), 
            },
        }
    else:
        raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}") 


    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                res = await response.json()
    except asyncio.TimeoutError:
        raise RuntimeError("request timed out; check the server or network connection.")
    except Exception as e:
        raise RuntimeError(f"request failed: {e}")

    if forward_only or res.get("forward_only"):
        meta_info = res.get("meta_info", {})
        if "scores" in res:
            scores = res["scores"]
        else:
            scores = meta_info.get("scores", [])
        return scores, meta_info
    else:
        try:
            return res["text"], res["meta_info"]
        except Exception as e:
            if forward_only:
                return [], res["error"]
            else:
                return "", res["error"]


if __name__ == "__main__":
    import asyncio
    base_url = "http://localhost:12345"
    text = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$ Step 1: In polar coordinates, a point $(r, \\theta)$ represents the point that is $r$ units away from the origin and forms an angle of $\\theta$ with the positive $x$-axis. \u043a\u0438\n"
    max_step_tokens = 0
    stop = ["\n"]
    temperature = 0.7

    result = asyncio.run(generate_async(text, base_url, max_step_tokens=1024, stop=stop, temperature=temperature))
    print(result)
