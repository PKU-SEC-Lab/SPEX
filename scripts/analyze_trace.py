#!/usr/bin/env python3
"""Analyze rebase_async trace JSONs for intra-query speculation behavior."""
import json
import os
import sys
import glob
import statistics
from collections import defaultdict, Counter


def load_run(trace_dir, label):
    runs = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "q*_trace.json"))):
        with open(path) as f:
            runs.append((os.path.basename(path), json.load(f)))
    return label, runs


def per_query_metrics(label, runs):
    print(f"\n========== {label} ==========")
    print(f"{'qid':>5} {'nodes':>6} {'main':>5} {'spec':>5} "
          f"{'leaves':>6} {'spec_leaves':>11} "
          f"{'spec_useful':>11} {'spec_wasted':>11} "
          f"{'depth':>5} {'wallt':>7}")

    totals = defaultdict(int)
    per_depth_main = defaultdict(int)
    per_depth_spec = defaultdict(int)
    per_depth_spec_wasted = defaultdict(int)

    for path, data in runs:
        nodes = data["nodes"]
        spec = bool(data["speculative"])
        width = data["width"]
        # Index nodes by id and by parent
        children_of = defaultdict(list)
        node_by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            if n["parent_id"] is not None:
                children_of[n["parent_id"]].append(n)

        n_main = sum(1 for n in nodes if n["created_via"] == "main")
        n_spec = sum(1 for n in nodes if n["created_via"] == "spec")
        leaves = sum(1 for n in nodes if n["is_leaf"])
        spec_leaves = sum(1 for n in nodes if n["is_leaf"] and n["created_via"] == "spec")

        # A spec node is "useful" if either (a) it became a leaf that contributed an answer,
        # or (b) its parent was selected for further expansion AND this child is on the active main path
        # (i.e., the node itself is was_main_chosen=True meaning its own children were further expanded).
        # We'll define useful as: spec node that was either picked as a parent for the next layer
        # OR ended up as a leaf with an answer.
        spec_useful = 0
        spec_wasted = 0
        for n in nodes:
            if n["created_via"] != "spec":
                continue
            useful = n["was_main_chosen"] or (n["is_leaf"] and not n["dead"])
            if useful:
                spec_useful += 1
            else:
                spec_wasted += 1

        max_depth = max(n["depth"] for n in nodes)
        max_finish = max(n["finish_time"] for n in nodes)
        qid = path.replace("q", "").replace("_trace.json", "")
        print(f"{qid:>5} {len(nodes):>6} {n_main:>5} {n_spec:>5} "
              f"{leaves:>6} {spec_leaves:>11} "
              f"{spec_useful:>11} {spec_wasted:>11} "
              f"{max_depth:>5} {max_finish:>7.2f}")

        totals["nodes"] += len(nodes)
        totals["main"] += n_main
        totals["spec"] += n_spec
        totals["spec_useful"] += spec_useful
        totals["spec_wasted"] += spec_wasted
        totals["leaves"] += leaves
        totals["spec_leaves"] += spec_leaves

        for n in nodes:
            d = n["depth"]
            if n["created_via"] == "main":
                per_depth_main[d] += 1
            else:
                per_depth_spec[d] += 1
                if not (n["was_main_chosen"] or (n["is_leaf"] and not n["dead"])):
                    per_depth_spec_wasted[d] += 1

    print(f"\n--- {label} totals ---")
    print(f"  total nodes:          {totals['nodes']}")
    print(f"  main nodes:           {totals['main']}")
    print(f"  spec nodes:           {totals['spec']}")
    if totals['nodes']:
        print(f"  spec / total:         {totals['spec']/totals['nodes']:.2%}")
    print(f"  spec useful:          {totals['spec_useful']}")
    print(f"  spec wasted:          {totals['spec_wasted']}")
    if totals['spec']:
        print(f"  spec waste rate:      {totals['spec_wasted']/totals['spec']:.2%}")
    print(f"  leaves total:         {totals['leaves']}")
    print(f"  leaves from spec:     {totals['spec_leaves']}")

    print(f"\n  per-depth main / spec / spec_wasted:")
    all_depths = sorted(set(list(per_depth_main) + list(per_depth_spec)))
    for d in all_depths:
        m = per_depth_main.get(d, 0)
        s = per_depth_spec.get(d, 0)
        sw = per_depth_spec_wasted.get(d, 0)
        ratio = (sw/s*100) if s else 0
        print(f"    depth {d:>2}: main={m:>4} spec={s:>4} wasted={sw:>4} ({ratio:5.1f}% of spec)")

    return totals


def compare_timing(label, runs):
    """Look at finish_time distribution by depth, measure 'gap' between layers."""
    print(f"\n--- {label} per-query depth timing ---")
    print(f"{'qid':>4} | depth-by-depth median finish_time (s)")
    for path, data in runs:
        nodes = data["nodes"]
        per_depth = defaultdict(list)
        for n in nodes:
            per_depth[n["depth"]].append(n["finish_time"])
        qid = path.replace("q", "").replace("_trace.json", "")
        depths = sorted(per_depth)
        meds = [statistics.median(per_depth[d]) for d in depths]
        deltas = [meds[i] - meds[i-1] for i in range(1, len(meds))]
        line = " | ".join(f"d{d}:{meds[i]:5.1f}" for i, d in enumerate(depths))
        gaps = " ".join(f"{g:5.1f}" for g in deltas)
        print(f"{qid:>4} | {line}")
        print(f"      | layer gaps: {gaps}")


def main():
    if len(sys.argv) < 3:
        print("usage: analyze_trace.py BASE_DIR SPEC_DIR")
        sys.exit(1)
    base_dir, spec_dir = sys.argv[1], sys.argv[2]
    base_label, base_runs = load_run(base_dir, "BASELINE")
    spec_label, spec_runs = load_run(spec_dir, "SPECULATIVE")

    per_query_metrics(base_label, base_runs)
    per_query_metrics(spec_label, spec_runs)

    compare_timing(spec_label, spec_runs)


if __name__ == "__main__":
    main()
