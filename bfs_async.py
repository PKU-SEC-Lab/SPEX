import argparse
import json
import re
import time
import asyncio
from sglang import function, gen, RuntimeEndpoint
import fcntl
import os
import math
import yaml
import torch
import torch.nn.functional as F
import threading
from utils import generate_async, now_ts
from collections import defaultdict
from math_evaluate import extract_shepherd_answer, grade_answer, agg_min, agg_prod, agg_mean, agg_last
from utils import TreeNode, Tree, State
from evaluate.evaluate_utils.grader import *

ANS_RE = re.compile(r"The answer is: (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
GT_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
SHEPHERD_RE = re.compile(r"The answer is:(.+) \u043a\u0438")
METAMATH_RE = re.compile(r"The answer is:(.+)\n\n")


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




class BFSTreeNode(TreeNode):
    def __init__(self, id, state, score, curr_text, num_step_tokens=0, parent=None):
        super().__init__(id, state, score, curr_text, num_step_tokens, parent)
        self.theoretical_num_children = 0
        self.existing_num_children = 0
        self.n_pre_running = 0
        self.dead = False
        self.finish_time: float = 0.0
        self.answer: str | None = None
    
    def set_dead(self):
        self.dead = True
        for child in self.children:
            child.set_dead()

class BFSTree(Tree):
    def __init__(self, root_state, paras, reward_backend, ets):
        super().__init__(root_state, paras, reward_backend, BFSTreeNode)
        self.ets = ets
        # Per-tree useful-spec EMA (used by SPEC_USEFUL_GATE). Init = useful_init
        # so that early layers don't get throttled before any signal arrives.
        self.spec_useful_ema = float(paras.get("spec_useful_init", 0.5))
        self.spec_useful_samples = 0
        self.last_main_depth = 0

    def _trace_meta(self):
        m = super()._trace_meta()
        m["width"] = int(self.paras.get("width", 0))
        return m

    def _trace_node_dict(self, node):
        d = super()._trace_node_dict(node)
        d.update({
            "was_main_chosen": bool(getattr(node, "was_main_chosen", False)),
            "n_pre_running_final": int(getattr(node, "n_pre_running", 0)),
            "theoretical_num_children": int(getattr(node, "theoretical_num_children", 0)),
            "existing_num_children": int(getattr(node, "existing_num_children", 0)),
            "dead": bool(getattr(node, "dead", False)),
        })
        return d

    def insert(self, state, parent):
        if state.scores() == [] or state.scores is None:
            return
        score = state.scores()[-1]
        num_step_tokens = state.get_meta_info("step")["completion_tokens"]
        if parent is None:
            curr_text = state.text()
        else:
            parent_text_len = len(parent.get_text())
            curr_text = state.text()[parent_text_len:]
        new_node = BFSTreeNode(self.size_, state, score, curr_text, num_step_tokens, parent)
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
    
    async def select_and_expand(self, depth):
        cand_node_list = []
        cand_node_weights = []
        nodes = self.depth_nodes[depth]
        nodes.sort(key=lambda x:  x.get_score(), reverse=True)
        for node in self.depth_nodes[depth]:
            if node in self.leaf_nodes:
                self.leaf_nodes.remove(node)
            if node.dead == True:
                continue
            if node.parent is not None:
                if node.parent.theoretical_num_children <= node.existing_num_children:
                    continue
                else:
                    node.parent.existing_num_children += 1
            if node.is_leaf() == True or node.get_cum_tokens() >= self.paras["max_tokens"]:
                self.leaf_nodes.append(node)
                self.spec_leaf_nodes.remove(node)
                self.remaining_width -= 1              
            else:
                cand_node_list.append(node)
                cand_node_weights.append(node.get_score())
        if self.remaining_width <= 0 or cand_node_list == []:
            return False
        if self.ets:
            self.paras["select_method"] = "softmax_costmodel"
        if self.paras["select_method"] ==  "softmax":
            nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, self.remaining_width)
        if self.paras["select_method"] == "softmax_with_truncate":
            nodes, widths = self.select_softmax_with_truncation(cand_node_list, cand_node_weights, self.remaining_width)
        if self.paras["select_method"] == "softmax_costmodel":
            nodes, widths = self.select_softmax_costmodel(cand_node_list, cand_node_weights, self.remaining_width, depth)
        # --- Useful-spec EMA: count, among the candidates we *just chose to keep*, how
        # many were originally born from speculation. This is the realised hit-rate of
        # speculation at this layer; we feed it into a per-query EMA and use it to
        # throttle subsequent spec layers.
        if self.paras.get("spec_useful_gate", 0):
            spec_cands = [n for n in cand_node_list if getattr(n, "created_via", "main") == "spec"]
            if spec_cands:
                kept = sum(1 for n, w in zip(nodes, widths) if w > 0 and getattr(n, "created_via", "main") == "spec")
                layer_useful = kept / len(spec_cands)
                a = float(self.paras.get("spec_useful_alpha", 0.4))
                self.spec_useful_ema = (1 - a) * self.spec_useful_ema + a * layer_useful
                self.spec_useful_samples += 1
        temp = []
        while not self.request_queue.empty():
            temp.append(await self.request_queue.get())
        for expand_node, width in zip(nodes, widths):
            expand_node.theoretical_num_children = width
            if width > 0:
                expand_node.was_main_chosen = True
            if width <= 0:
                expand_node.set_dead()
            if width - expand_node.n_pre_running < 0:
                self.running_nodes[expand_node] += width - expand_node.n_pre_running
            elif width - expand_node.n_pre_running > 0:
                for _ in range(width - expand_node.n_pre_running):
                    self.running_nodes[expand_node]+=1
                    if self.budget_coordinator is not None:
                        self.budget_coordinator.claim("main")
                    await self.request_queue.put((expand_node, "main", now_ts()))
        # remember main wave-front depth so depth-decay knows how far ahead each spec layer is
        self.last_main_depth = depth
        # Re-check the spec-node candidates we set aside above: keep the still-live ones
        # in the queue, and drop any that became dead during main-path resolution.
        for item in temp:
            if isinstance(item, tuple):
                if len(item) == 3:
                    node_obj, source, t_enq = item
                elif len(item) == 2:
                    node_obj, source = item
                    t_enq = now_ts()
                else:
                    node_obj, source, t_enq = item[0], "main", now_ts()
            else:
                node_obj, source, t_enq = item, "main", now_ts()
            if not node_obj.dead:
                await self.request_queue.put((node_obj, source, t_enq))
            elif node_obj not in nodes:
                # Stray speculative item from an earlier layer that has since been
                # discarded — release its budget claim to avoid a slow leak.
                self.running_nodes[node_obj] = 0
                if self.budget_coordinator is not None:
                    self.budget_coordinator.release(source)
        return True 
    
    async def spec_all(self, depth):
        lookahead = max(1, int(self.paras.get("spec_lookahead", 2)))
        for d in range(depth + 1, depth + 1 + lookahead):
            if d >= len(self.depth_nodes):
                break
            if self.running_producers + self.remaining_width >= self.num_producers:
                break
            await self.speculative_expand(d)
    
    async def speculative_expand(self, depth):
        existing_nodes = self.depth_nodes[depth]
        cand_node_list = []
        cand_node_weights = []
        for node in existing_nodes:
            if node.dead == True or node.is_leaf() == True or node.get_cum_tokens() >= self.paras["max_tokens"]:
                continue
            cand_node_list.append(node)
            cand_node_weights.append(node.get_score())
        if not cand_node_list:
            return
        # --- top-K score gating: keep only the highest-scoring fraction of candidates
        top_k_frac = float(self.paras.get("spec_top_k_frac", 1.0))
        if top_k_frac < 1.0 - 1e-9:
            paired = sorted(zip(cand_node_list, cand_node_weights), key=lambda p: p[1], reverse=True)
            keep = max(1, math.ceil(top_k_frac * len(paired)))
            paired = paired[:keep]
            cand_node_list = [p[0] for p in paired]
            cand_node_weights = [p[1] for p in paired]
        # --- per-layer spec budget cap, relative to width
        spec_budget_scale = float(self.paras.get("spec_budget_scale", 1.2))
        layer_cap_mult = float(self.paras.get("spec_layer_cap_mult", 1e9))  # 1e9 = unbounded (legacy default)
        # --- depth-decay: spec layers further from the main wave-front get a shrinking cap.
        # `depth` is the spec layer; `last_main_depth` is the main wave-front. The first
        # spec layer (d == last_main_depth+1) gets full cap (decay=1), each layer beyond
        # multiplies by `decay`. A decay of 1.0 (default) preserves legacy behavior.
        depth_decay = float(self.paras.get("spec_depth_decay", 1.0))
        ahead = max(0, depth - self.last_main_depth - 1)
        cap_decay_factor = depth_decay ** ahead if depth_decay < 1.0 - 1e-9 else 1.0
        # --- useful-spec EMA gate: shrink the cap on queries where speculation has been wasteful.
        if self.paras.get("spec_useful_gate", 0) and self.spec_useful_samples >= int(self.paras.get("spec_useful_warmup", 1)):
            min_thr = float(self.paras.get("spec_useful_min", 0.25))
            min_factor = float(self.paras.get("spec_useful_min_factor", 0.25))
            if self.spec_useful_ema < min_thr:
                # linear interp from min_factor at ema=0 -> 1.0 at ema=min_thr
                gate_factor = min_factor + (1.0 - min_factor) * (self.spec_useful_ema / max(1e-6, min_thr))
            else:
                gate_factor = 1.0
        else:
            gate_factor = 1.0
        layer_cap = max(1, math.ceil(layer_cap_mult * self.paras["width"] * cap_decay_factor * gate_factor))
        spec_widths = math.ceil(spec_budget_scale * len(cand_node_list))
        spec_widths = min(spec_widths, layer_cap)
        # --- inter-query global cap: spec_count = total_budget - (main_active + spec_active)
        if self.budget_coordinator is not None:
            spec_widths = min(spec_widths, self.budget_coordinator.avail_spec())
        if spec_widths <= 0:
            return
        nodes, widths = self.select_softmax(cand_node_list, cand_node_weights, spec_widths)
        for node, width in zip(nodes, widths):
            existing_width = node.n_pre_running
            width_demand = width - existing_width
            if width_demand <= 0:
                continue
            if self.budget_coordinator is not None:
                # Atomically claim up to `width_demand` slots from the global budget.
                granted = self.budget_coordinator.try_claim_spec(width_demand)
                if granted <= 0:
                    continue
                width_demand = granted
            for _ in range(width_demand):
                self.running_nodes[node] += 1
                node.n_pre_running += 1
                await self.request_queue.put((node, "spec", now_ts()))

    

async def _reward_guided_search(s, id, question, ground_truth_answer, paras, reward_host, weight_agg, evaluation, ets, alpha):
    s = State(text=question, policy_url=s.stream_executor.backend.base_url, reward_url=reward_host.base_url)
    tree = BFSTree(s, paras, reward_host, ets)
    tree.query_id = id
    depth = 0
    while True:
        continue_search = await tree.select_and_expand(depth)
        if continue_search == False:
            break

        # Wait until all main-path expansions at this depth complete.
        while True:
            is_finished = True
            for node,running_num in tree.running_nodes.items():
                if node.get_depth() == depth and running_num > 0:
                    is_finished = False
            if is_finished:
                break

            try:
                finished_node, new_node = await asyncio.wait_for(tree.response_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                print(tree.running_producers)
                continue
            
            early_stop = False
            if paras["early_termination"]:
                if len(tree.leaf_nodes) > paras["width"]/4:
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
                    if len(max_scores) > 1:
                        max_scores.sort()
                        median_index = len(max_scores) // 2
                        if len(max_scores) % 2 == 0:
                            median_score = (max_scores[median_index - 1] + max_scores[median_index]) / 2
                        else:
                            median_score = max_scores[median_index]
                        if median_score > 0.9:
                            confident = True
                    if max_vote-second_vote > alpha*left*avg_score and confident:
                        early_stop=True
                        break
                        
            
            if tree.running_nodes[finished_node] <= 0:
                continue
            tree.running_nodes[finished_node] -= 1

            is_finished = True
            for node,running_num in tree.running_nodes.items():
                if node.get_depth() == depth and running_num > 0:
                    is_finished = False
            if is_finished:
                break
            if paras["speculative"] and new_node is not None and new_node.dead is False and new_node.is_leaf() == False:
                await tree.spec_all(depth)
        depth += 1

        if depth >= 25 or early_stop:
            break
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
        })
    for node in tree.leaf_nodes:
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
        real_leaf_nodes = [node for node in tree.leaf_nodes if node not in tree.spec_leaf_nodes]
        Tree.saving_rate.append(len(real_leaf_nodes)/paras["width"])
    elif evaluation=="greedy":
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)
        final_answer = answers[0]["answer"] if answers else None
        is_correct = grade_answer(final_answer, ground_truth_answer) if final_answer is not None else False
    Tree.correct_count += is_correct
    Tree.total_count += 1
    print(f"Question {id} finished, answer: {final_answer}, ground truth: {ground_truth_answer}, is correct: {is_correct}, total tokens: {total_tokens}")

    return is_correct




@function
def reward_guided_search(s, id, question, ground_truth_answer, paras, reward_host, agg_func, evaluation, ets, alpha):
    results = asyncio.run(
        _reward_guided_search(s, id, question, ground_truth_answer, paras, reward_host, agg_func, evaluation, ets, alpha)
    )
    return results
    

def main(args):
    if args.input_path.endswith(".jsonl"):
        prompts, test_examples = get_prompts(args)
    elif args.input_path.endswith(".json"):
        prompts, test_examples = get_prompts_list(args)
    else:
        raise ValueError(f"Unsupported input file: {args.input_path} (expected .json or .jsonl)")

    with open(args.parameter_path ,'r', encoding='utf-8') as file:
        paras = yaml.safe_load(file)
    
    paras["speculative"] = args.speculative
    if args.trace_dir:
        paras["trace_dir"] = args.trace_dir
    # Intra-query speculation tuning knobs (env-overridable; defaults preserve legacy behavior)
    if "SPEC_LOOKAHEAD" in os.environ:
        paras["spec_lookahead"] = int(os.environ["SPEC_LOOKAHEAD"])
    if "SPEC_BUDGET_SCALE" in os.environ:
        paras["spec_budget_scale"] = float(os.environ["SPEC_BUDGET_SCALE"])
    if "SPEC_LAYER_CAP_MULT" in os.environ:
        paras["spec_layer_cap_mult"] = float(os.environ["SPEC_LAYER_CAP_MULT"])
    if "SPEC_TOP_K_FRAC" in os.environ:
        paras["spec_top_k_frac"] = float(os.environ["SPEC_TOP_K_FRAC"])
    if "SPEC_DEPTH_DECAY" in os.environ:
        paras["spec_depth_decay"] = float(os.environ["SPEC_DEPTH_DECAY"])
    if "SPEC_USEFUL_GATE" in os.environ:
        paras["spec_useful_gate"] = int(os.environ["SPEC_USEFUL_GATE"])
    if "SPEC_USEFUL_MIN" in os.environ:
        paras["spec_useful_min"] = float(os.environ["SPEC_USEFUL_MIN"])
    if "SPEC_USEFUL_MIN_FACTOR" in os.environ:
        paras["spec_useful_min_factor"] = float(os.environ["SPEC_USEFUL_MIN_FACTOR"])
    if "SPEC_USEFUL_ALPHA" in os.environ:
        paras["spec_useful_alpha"] = float(os.environ["SPEC_USEFUL_ALPHA"])
    if "SPEC_USEFUL_INIT" in os.environ:
        paras["spec_useful_init"] = float(os.environ["SPEC_USEFUL_INIT"])
    if "SPEC_USEFUL_WARMUP" in os.environ:
        paras["spec_useful_warmup"] = int(os.environ["SPEC_USEFUL_WARMUP"])
    print(f"[spec-knobs] lookahead={paras.get('spec_lookahead', 2)} budget_scale={paras.get('spec_budget_scale', 1.2)} layer_cap_mult={paras.get('spec_layer_cap_mult', 1e9)} top_k_frac={paras.get('spec_top_k_frac', 1.0)} depth_decay={paras.get('spec_depth_decay', 1.0)} useful_gate={paras.get('spec_useful_gate', 0)} useful_min={paras.get('spec_useful_min', 0.25)} useful_min_factor={paras.get('spec_useful_min_factor', 0.25)}")
    # Inter-query (cross-tree) global speculation budget. See utils.setup_inter_query_budget.
    from utils import setup_inter_query_budget
    setup_inter_query_budget(paras, args, "bfs")
    paras["early_termination"] = args.early_termination
    if args.width > 0:
        paras["width"] = args.width
    if args.num_threads > 0:
        paras["num_threads"] = args.num_threads



    input_list_dict = []
    agg_func=args.weight_func
    evaluation=args.evaluation
    ets=args.ets
    alpha=args.alpha
    paras["ets"]=ets
    start_time = time.time()


    num_questions = args.max_questions if args.max_questions > 0 else 40

    for i, prompt in enumerate(prompts):
        if i > num_questions - 1:
            break
        input_list_dict.append({"id":i, "question":prompt, "ground_truth_answer":test_examples[i], "paras":paras, "reward_host":RuntimeEndpoint(args.reward_host), "agg_func":agg_func, "evaluation":evaluation, "ets":ets, "alpha":alpha})
    results = reward_guided_search.run_batch(input_list_dict, backend=RuntimeEndpoint(args.policy_host), num_threads=paras["num_threads"], progress_bar=True)
    end_time = time.time()
    print(f"accuracy: {Tree.correct_count} / {Tree.total_count} = {Tree.correct_count / Tree.total_count}")
    avg_saving_rate=sum(Tree.saving_rate)/len(Tree.saving_rate)
    print(f"avg saving rate: {avg_saving_rate}")
    throughput = num_questions / (end_time - start_time) * 60
    elapsed = end_time - start_time
    accuracy = Tree.correct_count / Tree.total_count if Tree.total_count > 0 else 0.0
    print(f"[summary] elapsed={elapsed:.2f}s throughput_per_min={throughput:.2f} accuracy={accuracy:.4f} num_questions={num_questions} speculative={args.speculative} early_termination={args.early_termination} ets={args.ets} num_threads={args.num_threads} width={args.width}")
    log_data = {
        "throughput": throughput,
        "elapsed": elapsed,
        "accuracy": accuracy,
        "num_questions": num_questions,
        "speculative": args.speculative,
        "early_termination": args.early_termination,
        "width": args.width,
        "ets": args.ets,
        "batch": args.num_threads,
        "avg_saving_rate": avg_saving_rate,
    }
    coord = paras.get("budget_coordinator")
    if coord is not None:
        from utils import print_budget_stats, dump_global_request_log
        s = print_budget_stats(coord)
        log_data["budget"] = s
        dump_global_request_log(coord, paras.get("trace_dir", ""),
                                num_threads=args.num_threads, width=args.width,
                                speculative=args.speculative, algo="bfs")
    log_file = os.path.join(args.output_path, "llemma7b_bfs_log.jsonl")
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")

    return


    

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--input_path', type=str, required=True)
    args_parser.add_argument('--output_path', type=str, required=True)
    args_parser.add_argument('--parameter_path', type=str, required=True)
    args_parser.add_argument('--policy_host', type=str)
    args_parser.add_argument('--reward_host', type=str)

    args_parser.add_argument('--speculative', action='store_true', default=False)
    args_parser.add_argument('--early_termination', action='store_true', default=False)
    args_parser.add_argument('--num_threads', type=int, default=0)
    args_parser.add_argument('--width', type=int, default=0)
    args_parser.add_argument('--ets', action='store_true', default=False)
    args_parser.add_argument('--weight_func', type=str, choices=["min", "prod", "mean", "last"])
    args_parser.add_argument('--evaluation', type=str, choices=["greedy", "majority"])
    args_parser.add_argument('--alpha', type=float, default=1.0)
    args_parser.add_argument('--trace_dir', type=str, default="")
    args_parser.add_argument('--max_questions', type=int, default=0,
                             help="Number of questions to evaluate (0 = use default of 40).")

    args = args_parser.parse_args()
    main(args)
    os._exit(0)