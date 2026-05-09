import argparse
import json
import time
import asyncio
from sglang import function, gen, RuntimeEndpoint
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
import yaml
import threading
from utils import generate_async, TreeNode, Tree, State, now_ts
from collections import defaultdict
from tqdm import tqdm
import math
from sglang.lang.ir import SglSamplingParams
from sglang.global_config import global_config
from math_evaluate import extract_shepherd_answer, grade_answer
import random
import json
from math_evaluate import extract_shepherd_answer, grade_answer, agg_min, agg_prod, agg_mean, agg_last
from utils import TreeNode, Tree, State
from evaluate.evaluate_utils.grader import *
import os

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_prompts(args):
    test_cases = read_jsonl(args.input_path)
    prompts = []
    for test in test_cases:
        prompts.append(test["problem"])
    return prompts, test_cases

def read_json_list(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def get_prompts_list(args):
    test_cases = read_json_list(args.input_path)
    prompts = []
    for test in test_cases:
        if "content" in test.keys():
            prompts.append(test["content"])
        elif "Question" in test.keys():
            prompts.append(test["Question"]) 
            test["answer"]=test["Correct Answer"]
        elif "problem" in test.keys():
            prompts.append(test["problem"]) 
        elif "question" in test.keys():
            prompts.append(test["question"]) 
    return prompts, test_cases



class DFSTreeNode(TreeNode):
    def __init__(self, id, state, score, curr_text, num_step_tokens=0, parent=None):
        super().__init__(id, state, score, curr_text, num_step_tokens, parent)
        self.theoretical_num_children = 0
        self.existing_num_children = 0
        self.explored_id = -1
    


class DFSTree(Tree):
    def __init__(self, root_state, paras, reward_backend,num_path=20, mcts_wid=4, prm=False, num_spec=2):
        super().__init__(root_state, paras, reward_backend, DFSTreeNode)
        self.mcts_wid = mcts_wid
        self.explored_nodes = set()
        self.pre_explored_nodes = set()
        self.N: Dict[DFSTreeNode, int] = defaultdict(
            lambda: 0
        )  # total visit count for each node
        self.Q: Dict[DFSTreeNode, float] = defaultdict(
            lambda: 0.0
        )  # total reward of each node
        self.reward: Dict[DFSTreeNode, list] = defaultdict(lambda: [None] * num_path)
        self.uct: Dict[DFSTreeNode, list] = defaultdict(lambda: [None] * num_path)
        self.weight_scheduler = "const"
        self.mcts_exploration_weight = 2.0
        self.max_depth_allowed = 5
        self.show_tree_expansion = False
        self.pre_hit_num = 0
        self.pre_num = 0
        self.therotical_total_pre = 0
        self.num_path = num_path
        self.exploring_nodes = set()
        self.decode_time = []
        self.expand_time = []
        self.pre_explored_lists = defaultdict(list)
        self.parent2children: Dict[DFSTreeNode, List[DFSTreeNode]] = defaultdict(list)
        self.prm = prm
        self.num_spec = num_spec+1
        self.num_producers = mcts_wid*(num_spec+1)
        self.producers = [
            asyncio.create_task(self.producer()) for _ in range(self.num_producers)
        ]
        self.num_expand: Dict[DFSTreeNode, int] = defaultdict(int)
        self.pre_explore_attempts = 0
        self.pre_explore_rejected_slots = 0
        self.pre_explore_granted_slots = 0
        self.pre_explore_skipped_no_budget = 0
        self.pre_explore_no_idle = 0

    def _trace_meta(self):
        m = super()._trace_meta()
        m.update({
            "mcts_wid": int(self.mcts_wid),
            "num_path": int(self.num_path),
            "num_spec": int(self.num_spec),
            "max_depth": int(self.max_depth_allowed),
            "pre_hit_num": int(self.pre_hit_num),
            "pre_num": int(self.pre_num),
            "pre_explore_attempts": int(self.pre_explore_attempts),
            "pre_explore_granted_slots": int(self.pre_explore_granted_slots),
            "pre_explore_rejected_slots": int(self.pre_explore_rejected_slots),
            "pre_explore_skipped_no_budget": int(self.pre_explore_skipped_no_budget),
            "pre_explore_no_idle": int(self.pre_explore_no_idle),
        })
        return m

    def _trace_node_dict(self, node):
        d = super()._trace_node_dict(node)
        d.update({
            "rollout_id": int(getattr(node, "rollout_id", -1)),
            "explored_id": int(getattr(node, "explored_id", -1)),
        })
        return d


    @property
    def n_idle_producers(self):
        return len(self.producers)//self.mcts_wid - len(self.exploring_nodes)

    def _pre_select_child(self, root, i_path, pre_num, node_path):
        pre_nodes = []
        node_pre_visited_time = defaultdict(int)
        node_pre_Q = defaultdict(int)
        node_pre_reward_time = defaultdict(int)
        for node in node_path:
            node_pre_visited_time[node] = 1

        while len(pre_nodes) < pre_num and i_path < self.num_path:
            node = root
            done = False
            node_path = [node]
            while not done and node not in self.exploring_nodes:
                next_node, is_leaf = self._select_child(
                    node, i_path, is_pre=True, node_pre_visited_time = node_pre_visited_time)
                node_pre_visited_time[next_node] += 1
                if node != next_node:
                    node_path.append(next_node)
                done = next_node.is_leaf() or next_node.depth>=self.max_depth_allowed
                if not done and is_leaf:
                    pre_nodes.append(next_node)
                    self.pre_explored_nodes.add(next_node)
                    self.exploring_nodes.add(next_node)
                    break
                node = next_node
                if done:
                    if next_node.get_score() is not None:
                        reward = next_node.get_score()
                    else:
                        reward = 0
                    self.pre_explored_nodes.add(next_node)
                    for node in reversed(node_path):
                        if self.prm:
                            index = node_path.index(node)
                            sub_branch = node_path[index:]
                            scores = [n.get_score() for n in sub_branch]
                            node_pre_Q[node] += sum(scores)/len(scores)
                        else:
                            node_pre_Q[node] += reward
                        node_pre_reward_time[node] += 1
            i_path += 1
        self.pre_num += len(pre_nodes)
        self.therotical_total_pre += pre_num
        return pre_nodes

    def _select_child(
        self, node: DFSTreeNode, rollout_id: int, is_pre: bool = False, node_pre_visited_time =  defaultdict(int), node_pre_Q = defaultdict(int), pre_id = 0, node_pre_reward_time = defaultdict(int) 
    ) -> Tuple[DFSTreeNode, bool]:

        # for select, if there is unexplored children, select it randomly. if all children nodes
        # have been explored, select UCB, a leaf node means it has no children, return True
        # when there is unexplored node

        if node not in self.parent2children.keys():
            return node, True


        pre_explored = [
            n for n in self.parent2children[node] if n in self.pre_explored_nodes and node_pre_visited_time[n] == 0
        ]

        if pre_explored:
            if self.prm:
                next_node = max(pre_explored, key=lambda n: n.score_)
            else:
                next_node = random.choice(pre_explored)
            if not is_pre:
                # Pre-explored node was promoted to the main path: spec hit.
                is_terminal=next_node.is_leaf() or next_node.depth>=self.max_depth_allowed
                if next_node.is_leaf():
                    self.leaf_nodes.append(next_node)
                if not is_terminal:
                    self.pre_hit_num += 1
                self.pre_explored_nodes.remove(next_node)
            return next_node, False


        # if there are children unexplored
        unexplored = [
            n for n in self.parent2children[node] if n not in self.explored_nodes and n not in self.pre_explored_nodes
        ]
        if unexplored:
            if self.prm:
                next_node = max(unexplored, key=lambda n: n.score_)
            else:
                next_node = random.choice(unexplored)
            return next_node, True

        # if all have been explord, from parent2children dict, select one node with highest UCB score



        # Get the list of children for the current node
        children = self.parent2children[node]

        # Compute UCT values for each child node
        uct_values = {
            n: self._compute_uct(parent_node=node, node=n, rollout_id=rollout_id, node_pre_visited_time=node_pre_visited_time, node_pre_Q = node_pre_Q,  node_pre_reward_time = node_pre_reward_time)
            for n in children
        }
        # Find the child with the maximum UCT value. Tie-break: when not in pre-mode,
        # prefer children that already have a pre-explored descendant (spec hit).
        if is_pre:
            next_node = max(uct_values, key=uct_values.get)
        else:
            next_node = max(
                uct_values,
                key=lambda n: (uct_values[n], self.has_pre_child(n))
            )

        return next_node, False
    
    def has_pre_child(self, node: DFSTreeNode):
        if node.children:
            if node in self.pre_explored_nodes:
                return True
            for child in node.children:
                if self.has_pre_child(child):
                    return True
            else:
                return False
        else:
            return node in self.pre_explored_nodes
        
    def _compute_uct(
        self, parent_node: DFSTreeNode, node: DFSTreeNode, rollout_id: int, node_pre_visited_time = defaultdict(int), node_pre_Q = defaultdict(int), node_pre_reward_time = defaultdict(int)
    ):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
            return 999
        weight = self._get_weight(rollout_id)
        return (self.Q[node] + node_pre_Q[node]) / (self.N[node] + node_pre_reward_time[node]) + weight * math.sqrt(
            math.log(self.N[parent_node] + node_pre_visited_time[parent_node]) / (self.N[node] + node_pre_visited_time[node])
        )

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.mcts_exploration_weight * (0.1 ** (rollout_id / self.num_path))
        elif self.weight_scheduler == "lin":
            return self.mcts_exploration_weight * (
                1 - 0.9 * (rollout_id / self.num_path)
            )
        elif self.weight_scheduler == "const":
            return self.mcts_exploration_weight
            
    def draw_tree(self):

        def display_tree(node: DFSTreeNode):
            print("|" + "-" * (node.depth * 4) + str(node) + "......" + str(node.get_score() if node.get_score() != None else self.Q[node]))
            for child in node.children:
                display_tree(child)

        print(f"---------Expanded Tree---------")
        display_tree(self.root_)
        print()
    
    def update_uct(self,i_path):
        
        def update_child(node: DFSTreeNode,i_path: int):
            self.uct[node][i_path]=self._compute_uct(parent_node=node.parent, node=node, rollout_id=i_path)
            for child in node.children:
                update_child(child, i_path)
        update_child(self.root_,i_path)



async def _dfs(s, id, question, ground_truth_answer, paras, reward_host,num_path=20,wid=4,prm=False,num_spec=2, weight_agg="min", evaluation="majority",alpha=0.5):
    s = State(text=question, policy_url=s.stream_executor.backend.base_url, reward_url=reward_host.base_url)
    tree = DFSTree(s, paras, reward_host, num_path, wid, prm,num_spec)
    tree.query_id = id
    tree.q_start_ts = now_ts()
    coord = paras.get("budget_coordinator", None)
    speculative = bool(paras.get("speculative", True))

    async def _enqueue_main(node):
        """Claim a 'main' budget slot then enqueue."""
        if coord is not None:
            coord.claim("main")
        await tree.request_queue.put((node, "main", now_ts()))

    async def _enqueue_spec_batch(spec_nodes_with_count):
        """Atomically claim spec slots up to global budget, then enqueue
        as many *full* (mcts_wid) batches as we can afford.

        spec_nodes_with_count: list[(pre_node, count)] where count = mcts_wid.
        Returns the list of pre_nodes actually enqueued (caller must purge
        the remaining ones from tree.exploring_nodes; otherwise the main
        path will think they are being explored and wait forever for
        responses that never come).
        """
        if not spec_nodes_with_count:
            return []
        total_slots = sum(c for _, c in spec_nodes_with_count)
        if coord is not None:
            granted = coord.try_claim_spec(total_slots)
            tree.pre_explore_granted_slots += granted
            tree.pre_explore_rejected_slots += (total_slots - granted)
            if granted <= 0:
                tree.pre_explore_skipped_no_budget += 1
                return []
            enqueued = []
            slots_left = granted
            for pre_node, count in spec_nodes_with_count:
                if slots_left >= count:
                    for _ in range(count):
                        await tree.request_queue.put((pre_node, "spec", now_ts()))
                    slots_left -= count
                    enqueued.append(pre_node)
                else:
                    break
            # Release any spec slots we claimed but didn't enqueue.
            for _ in range(slots_left):
                coord.release("spec")
            return enqueued
        else:
            # No coordinator: enqueue all (legacy behavior).
            tree.pre_explore_granted_slots += total_slots
            for pre_node, count in spec_nodes_with_count:
                for _ in range(count):
                    await tree.request_queue.put((pre_node, "spec", now_ts()))
            return [pn for pn, _ in spec_nodes_with_count]

    traj_list = []
    model_rollout_nodes = []
    n_explored_nodes = 0 
    for i_path in tqdm(range(num_path), desc=f"Running {num_path} MCTS paths"):
        node = tree.root_
        if node.is_leaf():
            print(f"Question {id} failed!")
            answer_for_the_question = {"id":id, "question": question, "model_answer":" ", "ground_truth_answer": ground_truth_answer["answer"], "total_tokens":0}
            return answer_for_the_question
        done = False
        node_path = [node]  # for boostrapping
        start = time.time()
        while not done:
            next_node, is_leaf = tree._select_child(
                node, i_path
            )
            if next_node in tree.exploring_nodes:
                is_leaf = True
            tree.explored_nodes.add(next_node)
            if node != next_node:
                node_path.append(next_node)
            done = next_node.is_leaf() or next_node.depth>=tree.max_depth_allowed
            if not done and is_leaf:
                next_node.explored_id = n_explored_nodes
                n_explored_nodes += 1
                async def pre_explore():
                    if not speculative:
                        return
                    pre_num = tree.n_idle_producers
                    tree.pre_explore_attempts += 1
                    if pre_num <= 0:
                        tree.pre_explore_no_idle += 1
                        return
                    if len(tree.root_.children) > 0 and pre_num > 0:
                        pre_nodes = tree._pre_select_child(tree.root_, i_path+1, pre_num, node_path)
                        # Build list of (pre_node, mcts_wid) batches, atomically claim spec budget.
                        batches = [(pre_node, tree.mcts_wid) for pre_node in pre_nodes]
                        enqueued = await _enqueue_spec_batch(batches)
                        # Critical: _pre_select_child already added every pre_node
                        # to tree.exploring_nodes.  For any pre_node we couldn't
                        # enqueue (budget rejected), we must remove it; otherwise
                        # the next main-path iteration that selects this node will
                        # treat it as "in flight" (is_leaf=True) and wait forever
                        # for responses that were never enqueued.
                        # NB: we use the node objects directly — `id` is shadowed
                        # by the question-id parameter of _dfs.
                        enqueued_set = set(enqueued)
                        for pre_node in pre_nodes:
                            if pre_node not in enqueued_set:
                                tree.exploring_nodes.discard(pre_node)

                if next_node not in tree.exploring_nodes:
                    tree.exploring_nodes.add(next_node)
                    for _ in range(tree.mcts_wid):
                        await _enqueue_main(next_node)
                    await pre_explore()

                while True:
                    try:
                        finished_node, child = await asyncio.wait_for(tree.response_queue.get(),timeout=1000)
                    except Exception as e:
                        print(f"error in response queue get: {e}")
                    
                    if child is not None: 
                        tree.parent2children[finished_node].append(child)
                        if finished_node == next_node and child.is_leaf():
                            tree.spec_leaf_nodes.remove(child)
                        if finished_node != next_node and child.is_leaf():
                            tree.leaf_nodes.remove(child)
                    elif tree.num_expand[finished_node]<2*tree.mcts_wid and tree.num_spec>1:
                        # Retry on failure.  Re-claim budget unconditionally
                        # (retries are already bounded by 2*mcts_wid per node;
                        # rejecting them would risk deadlocks since the inner
                        # loop relies on getting back exactly mcts_wid valid
                        # responses).  This is the "lenient" stance the DFS
                        # branch can afford because it is rarely compute-bound.
                        retry_source = "main" if finished_node == next_node else "spec"
                        if coord is not None:
                            coord.claim(retry_source)
                        await tree.request_queue.put((finished_node, retry_source, now_ts()))
                    else:
                        tree.parent2children[finished_node].append(child)
                    tree.num_expand[finished_node] += 1

                    if len(tree.parent2children[finished_node])==tree.mcts_wid:
                        node_children_clean = []
                        for child in tree.parent2children[finished_node]:
                            if child is not None:
                                node_children_clean.append(child)
                        tree.parent2children[finished_node] = node_children_clean
                        num_children = len(tree.parent2children[finished_node])
                        if num_children < tree.mcts_wid and tree.num_expand[finished_node]<2*tree.mcts_wid and tree.num_spec==1:
                            additional_expand = tree.mcts_wid-num_children
                            retry_source = "main" if finished_node == next_node else "spec"
                            if coord is not None:
                                for _ in range(additional_expand):
                                    coord.claim(retry_source)
                            for _ in range(additional_expand):
                                await tree.request_queue.put((finished_node, retry_source, now_ts()))
                        else:
                            assert tree.parent2children[finished_node] is not None
                            if tree.parent2children[finished_node]==[]:
                                finished_node.leaf_=True
                            # If the response is for next_node, the main rollout has finished
                            # one expansion; otherwise it was a pre-explored (speculative) node.
                            if finished_node == next_node:
                                next_node_children = tree.parent2children[next_node]
                                tree.exploring_nodes.remove(finished_node)
                                break
                            else:
                                tree.exploring_nodes.remove(finished_node)

                                await pre_explore()
                assert next_node_children is not None
                for c in next_node_children:
                    assert c is not None
                    c.rollout_id = i_path

            if tree.show_tree_expansion:
                tree.draw_tree()
            if done:
                tree.explored_nodes.add(next_node)   
            node = next_node  
            if node.leaf_:
                break
        else:
            end = time.time()      
            tree.decode_time.append(end-start)  
            reward = next_node.get_score()
            for node in reversed(node_path):
                if tree.prm:
                    index = node_path.index(node)
                    sub_branch = node_path[index:]
                    scores = [n.get_score() for n in sub_branch]
                    tree.Q[node] += sum(scores)/len(scores)
                    tree.reward[node][i_path]=(sum(scores)/len(scores))
                else:
                    tree.Q[node] += reward
                    tree.reward[node][i_path]=reward
                tree.N[node] += 1
                tree.explored_nodes.add(node)
        model_rollout_nodes.append(next_node)
        tree.pre_explored_lists[i_path] = [str(n) for n in tree.pre_explored_nodes]
        tree.update_uct(i_path)
        early_stop = False
        if i_path>num_path/2 and len(tree.leaf_nodes)>0:
            if paras["early_termination"]:
                answers=[]
                for node in tree.leaf_nodes:
                    step_scores = []
                    last_node = node
                    while last_node.get_depth() > 0:
                        step_scores.append(last_node.get_score())
                        last_node = last_node.get_parent()
                    step_scores.reverse()
                    answers.append({"answer":node.answer, "score":node.get_score(), "step_scores":step_scores})
                max_vote = 0
                max_rep = None
                equiv_classes = []
                equiv_weights = []
                equiv_all_scores = []
                equiv_num = []
                for cand in answers:
                    ans = cand["answer"]
                    weight = 1
                    if weight_agg == "min":
                        weight_func = agg_min
                    if weight_agg == "prod":
                        weight_func = agg_prod
                    if weight_agg == "mean":
                        weight_func = agg_mean
                    if weight_agg == "last":
                        weight_func = agg_last
                    weight = weight_func(cand["step_scores"])
                    flag = 0
                    
                    for i, rep in enumerate(equiv_classes):
                        if grade_answer(ans,rep):
                            flag = 1
                            equiv_weights[i] = equiv_weights[i]+weight
                            equiv_all_scores[i].append(weight)
                            equiv_num[i]+=1
                            if equiv_weights[i] > max_vote:
                                max_vote = equiv_weights[i]
                                max_rep = ans
                    if flag:
                        continue
                    equiv_classes.append(ans)
                    equiv_weights.append(weight)
                    equiv_all_scores.append([weight])
                    equiv_num.append(1)
                    if weight > max_vote:
                        max_vote = weight
                        max_rep = ans
                final_answer=str(max_rep)
                sun_w=sum(equiv_weights)
                equiv_weights_sorted=sorted(equiv_weights,reverse=True)
                second_vote = equiv_weights_sorted[1] if len(equiv_weights)>1 else 0
                second_num = equiv_num[equiv_weights.index(second_vote)] if second_vote>0 else 0
                max_index = equiv_weights.index(max_vote)
                max_scores = equiv_all_scores[max_index]
                max_num = equiv_num[equiv_weights.index(max_vote)] if max_vote>0 else 0

                left=paras["width"]-len(tree.leaf_nodes)
                max_avg_score = max_vote/max_num if max_num>0 else 0
                avg_score=second_vote/second_num if second_num>0 else 0
                
                confident = False
                if len(max_scores) > 2:
                    max_scores.sort()
                    median_index = len(max_scores) // 2
                    if len(max_scores) % 2 == 0:
                        median_score = (max_scores[median_index - 1] + max_scores[median_index]) / 2
                    else:
                        median_score = max_scores[median_index]
                    if median_score > 0.95:
                        confident = True
                if max_vote-second_vote > alpha*left*avg_score and confident:
                    early_stop=True
                    break
        
    # Per-query trace dump + producer cleanup must happen before we bail out.
    if paras.get("trace_dir"):
        try:
            os.makedirs(paras["trace_dir"], exist_ok=True)
            tree.dump_trace(os.path.join(paras["trace_dir"], f"q{id}_trace.json"))
        except Exception as _e:
            print(f"[trace dump error] q{id}: {_e}")
    await tree.shutdown()
    total_tokens = 0
    all_nodes = tree.get_nodes()
    answers = []
    nodes_info = []
    answer_store_path = paras["store_path"] + f"answer_q{id}.json"
    for node in all_nodes:
        total_tokens += node.num_step_tokens
        if node.get_parent() is not None:
            parent_id = node.get_parent().get_id() 
        else:
            parent_id = None
        nodes_info.append({
            "id": node.get_id(),
            "text": node.curr_text,
            "score": node.get_score(),
            "parent_id": parent_id,
            "depth": node.get_depth(),
            "leaf": node.is_leaf(),
            "num_tokens": node.num_step_tokens,
            "explored_id": node.explored_id,
        })
        if node.is_leaf():
            answer = extract_shepherd_answer(node.get_text())
            correct = grade_answer(answer, ground_truth_answer["answer"])
            nodes_info[-1]["answer"] = answer
            nodes_info[-1]["correct"] = correct
        if node.is_leaf():
            step_scores = []
            last_node = node
            while last_node.get_depth() > 0:
                step_scores.append(last_node.get_score())
                last_node = last_node.get_parent()
            step_scores.reverse()
            answers.append({"answer":node.answer, "score":node.get_score(), "step_scores":step_scores})
    for node in tree.spec_leaf_nodes:
        if node in tree.leaf_nodes:
            continue
        step_scores = []
        last_node = node
        while last_node.get_depth() > 0:
            step_scores.append(last_node.get_score())
            last_node = last_node.get_parent()
        step_scores.reverse()
        answers.append({"answer":node.answer, "score":node.get_score(), "step_scores":step_scores})
    text = ground_truth_answer["answer"]
    if '\n####' in text:
        parts = text.split('\n####')

        ground_truth_answer=parts[-1]
    else:
        ground_truth_answer=text

    if evaluation=="majority":
        max_vote = 0
        max_rep = None
        equiv_classes = []
        equiv_weights = []
        for cand in answers:
            ans = cand["answer"]
            weight = 1
            if weight_agg == "min":
                weight_func = agg_min
            if weight_agg == "prod":
                weight_func = agg_prod
            if weight_agg == "mean":
                weight_func = agg_mean
            if weight_agg == "last":
                weight_func = agg_last
            weight = weight_func(cand["step_scores"])
            flag = 0
            
            for i, rep in enumerate(equiv_classes):
                if grade_answer(ans,rep):
                    flag = 1
                    equiv_weights[i] = equiv_weights[i]+weight
                    if equiv_weights[i] > max_vote:
                        max_vote = equiv_weights[i]
                        max_rep = ans
            if flag:
                continue
            equiv_classes.append(ans)
            equiv_weights.append(weight)
            if weight > max_vote:
                max_vote = weight
                max_rep = ans
        final_answer=str(max_rep)
        is_correct = grade_answer(str(max_rep), ground_truth_answer) if final_answer is not None else False
        Tree.saving_rate.append(i_path/num_path)
        print(f'saving rate: {i_path/num_path}')
    elif evaluation=="greedy":
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        final_answer = answers[0]["answer"] if answers else None
        is_correct = grade_answer(final_answer, ground_truth_answer) if final_answer is not None else False
    Tree.correct_count += is_correct
    Tree.total_count += 1
    
    print(f"Question {id} finished, answer: {final_answer}, ground truth: {ground_truth_answer}, is correct: {is_correct}, total tokens: {total_tokens}")

    answer_for_the_question = {"id":id, "question": question, "model_answer":answers, "ground_truth_answer": ground_truth_answer, "total_tokens":total_tokens}
    return answer_for_the_question


def tree_to_json(tree:DFSTree):
    """Serialise a DFSTree into a nested dict suitable for JSON dump."""

    def node_to_dict(node:DFSTreeNode):
        # Forward-fill rewards once a path has produced one (so JSON keeps
        # monotonic per-rollout reward arrays for downstream analysis).
        gen=False
        for i in range(len(tree.reward[node])):
            if (not gen) and tree.reward[node][i] is not None:
                gen=True
            if gen and tree.reward[node][i] is None:
                tree.reward[node][i]=tree.reward[node][i-1]
        gen=False
        if node not in tree.parent2children:

            return {
                "id": node.id,
                "depth": node.depth,
                "N": tree.N[node],
                "Q": tree.Q[node],
                "reward": tree.reward[node],
                "uct": tree.uct[node],
                "text": node.text_,
                "children": []
            }

        children_list = []
        for child in tree.parent2children[node]:
            children_list.append(node_to_dict(child))
        return {
            "id": node.id,
            "depth": node.depth,
            "N": tree.N[node],
            "Q": tree.Q[node],
            "reward": tree.reward[node],
            "uct": tree.uct[node],
            "text": node.text_,
            "children": children_list
        }

    tree_dict = {
        "root": node_to_dict(tree.root_),
        "properties": {
            "mcts_wid": tree.mcts_wid,
            "num_path": tree.num_path,
            "exploration_weight": tree.mcts_exploration_weight,
            "max_depth": tree.max_depth_allowed,
            "prm_enabled": tree.prm,
            "num_producers": tree.num_producers
        }
    }
    return tree_dict

@function
def dfs(s, id, question, ground_truth_answer, paras, reward_host, num_path, wid, prm, num_spec, agg_func, evaluation):
    asyncio.run(
        _dfs(s, id, question, ground_truth_answer, paras, reward_host, num_path, wid, prm, num_spec, agg_func, evaluation)
    )
    
def main(args):
    if args.input_path.endswith(".jsonl"):
        prompts, test_examples = get_prompts(args)
    elif args.input_path.endswith(".json"):
        prompts, test_examples = get_prompts_list(args)
    else:
        raise ValueError(f"Unsupported input file: {args.input_path} (expected .json or .jsonl)")

    with open(args.parameter_path ,'r', encoding='utf-8') as file:
        paras = yaml.safe_load(file)
    input_list_dict = []
    num_path=args.num_path
    if args.num_threads > 0:
        paras["num_threads"] = args.num_threads
    wid=args.wid
    print(args.rm)
    prm=(args.rm=="prm")

        
    num_spec = args.num_spec
    agg_func=args.weight_func
    evaluation=args.evaluation
    paras["early_termination"] = args.early_termination

    # ---- Speculative / trace / inter-query budget setup (mirrors bfs_async) ----
    paras["speculative"] = args.speculative
    if args.trace_dir:
        paras["trace_dir"] = args.trace_dir
    # MCTS-specific intra-query knobs (env-overridable). Currently just a global
    # cap on pre_explore aggressiveness; defaults preserve legacy behavior.
    if "DFS_PRE_FRAC" in os.environ:
        paras["dfs_pre_frac"] = float(os.environ["DFS_PRE_FRAC"])
    if "DFS_PRE_MAX" in os.environ:
        paras["dfs_pre_max"] = int(os.environ["DFS_PRE_MAX"])
    print(f"[dfs-knobs] speculative={paras.get('speculative', True)} pre_frac={paras.get('dfs_pre_frac', 1.0)} pre_max={paras.get('dfs_pre_max', -1)}")
    # Inter-query (cross-tree) global speculation budget.
    # Same env var as bfs_async to keep tooling unified.
    from utils import setup_inter_query_budget
    setup_inter_query_budget(paras, args, "dfs")

    num_questions = args.max_questions if args.max_questions > 0 else 40

    start_time = time.time()
    for i, prompt in enumerate(prompts):
        if i >= num_questions:
            break
        input_list_dict.append({"id":i, "question":prompt, "ground_truth_answer":test_examples[i], "paras":paras, "reward_host":RuntimeEndpoint(args.reward_host), "num_path":num_path, "wid":wid, "prm":prm, "num_spec":num_spec, "agg_func":agg_func, "evaluation":evaluation})

    states = dfs.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=paras["num_threads"], progress_bar=True)
    end_time = time.time()
    print(f"accuracy: {Tree.correct_count} / {Tree.total_count} = {Tree.correct_count / Tree.total_count}")
    avg_saving_rate=sum(Tree.saving_rate)/len(Tree.saving_rate)
    print(f"avg saving rate: {avg_saving_rate}")
    elapsed = end_time - start_time
    accuracy = Tree.correct_count / Tree.total_count if Tree.total_count > 0 else 0.0
    throughput = num_questions / elapsed * 60 if elapsed > 0 else 0.0
    print(f"[summary] elapsed={elapsed:.2f}s throughput_per_min={throughput:.2f} accuracy={accuracy:.4f} num_questions={num_questions} speculative={args.speculative} early_termination={args.early_termination} num_threads={args.num_threads} wid={args.wid} num_spec={args.num_spec} num_path={num_path}")
    log_data = {
        "accuracy": accuracy,
        "time": elapsed,
        "throughput": throughput,
        "num_questions": num_questions,
        "num_path": num_path,
        "prm": prm,
        "speculative": args.speculative,
        "early_termination": args.early_termination,
        "num_threads": args.num_threads,
        "wid": args.wid,
        "num_spec": args.num_spec,
        "avg_saving_rate": avg_saving_rate,
    }
    coord = paras.get("budget_coordinator")
    if coord is not None:
        from utils import print_budget_stats, dump_global_request_log
        s = print_budget_stats(coord)
        log_data["budget"] = s
        dump_global_request_log(coord, paras.get("trace_dir", ""),
                                num_threads=args.num_threads, width=args.wid,
                                speculative=args.speculative, algo="dfs")
    log_file = os.path.join(args.output_path, "llemma7b_dfs_log.jsonl")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")
    results = []
    total_gen_tokens = 0
    for s in states:
        answer = s.ret_value
        results.append(answer)


    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--parameter_path', type=str, required=True)
    args_parser.add_argument('--policy_host', type=str)
    args_parser.add_argument('--reward_host', type=str)
    args_parser.add_argument('--num_path', type=int, default=20)
    args_parser.add_argument('--wid', type=int, default=4)
    args_parser.add_argument('--rm', type=str, choices=["orm", "prm"])
    args_parser.add_argument('--num_spec', type=int, default=2)
    args_parser.add_argument('--weight_func', type=str, choices=["min", "prod", "mean", "last"])
    args_parser.add_argument('--evaluation', type=str, choices=["greedy", "majority"])
    args_parser.add_argument('--num_threads', type=int, default=0)
    args_parser.add_argument('--early_termination', action='store_true', default=False)
    args_parser.add_argument('--speculative', action='store_true', default=False,
                             help="Enable pre-explore (intra-query speculation). Off by default for clean baselines.")
    args_parser.add_argument('--trace_dir', type=str, default="",
                             help="If set, dump per-query traces and global request log to this directory.")
    args_parser.add_argument('--max_questions', type=int, default=0,
                             help="Override the auto-selected number of questions (0 = use default by num_threads).")
    args = args_parser.parse_args()
    main(args)
    os._exit(0)