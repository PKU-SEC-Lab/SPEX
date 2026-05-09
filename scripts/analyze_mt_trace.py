#!/usr/bin/env python3
"""
Multi-thread per-request trace analyzer.

Reads `<TRACE_DIR>/global_requests.json` (timeline of every policy/reward
generation request, each tagged with query_id, source=main|spec, depth and
three timestamps) plus the per-query node files `qN_trace.json` (tells us
whether each spec leaf was eventually picked by `select_and_expand` --
"useful" -- or wasted).

Outputs:

  1. Per-request table (start/end times) -- written to
     <TRACE_DIR>/global_requests.csv (and a head sample is printed).
  2. Aggregate stats: per-query elapsed, request-mix, queue-wait, gen
     latency, useful-spec ratio.
  3. Concurrency-over-time profile: at every event boundary, how many
     `main` / `spec` requests are in flight. From this we derive:
        - mean / max in-flight
        - %% of wall-clock time the policy server is "saturated" (#main >= cap)
        - %% of wall-clock time idle slots exist that *could* host more spec
          but didn't (slack)
  4. Cross-query alignment: for each pair of overlapping queries, when
     do their layer transitions happen?  This shows whether the budget
     coordinator is starving one query while the other is between layers.

Run:
  python3 scripts/analyze_mt_trace.py exp_results/mt_trace/traces
"""
import json
import os
import sys
import csv
from collections import defaultdict


def load(trace_dir):
    g = json.load(open(os.path.join(trace_dir, "global_requests.json")))
    per_query = {}
    for fn in os.listdir(trace_dir):
        if fn.startswith("q") and fn.endswith("_trace.json"):
            try:
                qid = int(fn[1:].split("_")[0])
            except ValueError:
                continue
            per_query[qid] = json.load(open(os.path.join(trace_dir, fn)))
    return g, per_query


def fmt(t):
    return f"{t:7.3f}"


def per_request_table(g, out_csv, head=20):
    reqs = sorted(g["requests"], key=lambda r: r["t_enqueue"])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "query_id", "source", "depth", "parent_id", "child_id",
            "t_enqueue", "t_pickup", "t_done",
            "queue_wait", "gen_latency", "ok",
        ])
        for r in reqs:
            qw = r["t_pickup"] - r["t_enqueue"]
            gl = r["t_done"] - r["t_pickup"]
            w.writerow([
                r["query_id"], r["source"], r["depth"], r["parent_id"], r["child_id"],
                f"{r['t_enqueue']:.4f}", f"{r['t_pickup']:.4f}", f"{r['t_done']:.4f}",
                f"{qw:.4f}", f"{gl:.4f}", int(bool(r["ok"])),
            ])
    print(f"[csv]  per-request table written: {out_csv} ({len(reqs)} rows)\n")

    print("Sample (first {} requests, sorted by enqueue):".format(head))
    print(f"  {'q':>2} {'src':>4} {'d':>3} {'parent':>6} {'child':>6} "
          f"{'enq':>8} {'pickup':>8} {'done':>8} {'wait':>6} {'gen':>6}")
    for r in reqs[:head]:
        qw = r["t_pickup"] - r["t_enqueue"]
        gl = r["t_done"] - r["t_pickup"]
        print(f"  {r['query_id']:>2} {r['source']:>4} {r['depth']:>3} "
              f"{r['parent_id']:>6} {r['child_id']:>6} "
              f"{fmt(r['t_enqueue'])} {fmt(r['t_pickup'])} {fmt(r['t_done'])} "
              f"{qw:6.3f} {gl:6.3f}")
    print()


def per_query_summary(g, per_query):
    reqs = g["requests"]
    by_q = defaultdict(list)
    for r in reqs:
        by_q[r["query_id"]].append(r)

    # cross-ref usefulness from node trace
    chosen_by_q = {}
    start_offset = {}  # per-query t0 offset (q_trace records use per-query epoch)
    for qid, td in per_query.items():
        nodes = td.get("nodes", []) if isinstance(td, dict) else td
        chosen_by_q[qid] = {n["id"] for n in nodes if n.get("was_main_chosen")}
        start_offset[qid] = td.get("start_time", 0.0) if isinstance(td, dict) else 0.0

    print("Per-query summary:")
    print(f"  {'q':>2} {'start':>7} {'end':>7} {'elapsed':>7} {'#main':>6} "
          f"{'#spec':>6} {'spec_useful':>11} {'qwait_avg':>9} {'gen_avg':>7}")
    rows = []
    for qid in sorted(by_q):
        rs = by_q[qid]
        t0 = min(r["t_enqueue"] for r in rs)
        t1 = max(r["t_done"] for r in rs)
        n_main = sum(1 for r in rs if r["source"] == "main")
        n_spec = sum(1 for r in rs if r["source"] == "spec")
        chosen = chosen_by_q.get(qid, set())
        spec_used = sum(1 for r in rs
                        if r["source"] == "spec" and r["ok"] and r["child_id"] in chosen)
        qw = sum(r["t_pickup"] - r["t_enqueue"] for r in rs) / len(rs)
        gl = sum(r["t_done"] - r["t_pickup"] for r in rs) / len(rs)
        useful_pct = 100.0 * spec_used / n_spec if n_spec else 0.0
        rows.append((qid, t0, t1, t1 - t0, n_main, n_spec, useful_pct, qw, gl))
        print(f"  {qid:>2} {t0:7.2f} {t1:7.2f} {t1-t0:7.2f} {n_main:>6} {n_spec:>6} "
              f"{useful_pct:9.1f}%  {qw:8.3f} {gl:6.3f}")
    print()
    return rows


def write_query_csv(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "start", "end", "elapsed",
                    "n_main", "n_spec", "spec_useful_pct",
                    "queue_wait_avg", "gen_latency_avg"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                        r[4], r[5], f"{r[6]:.2f}", f"{r[7]:.4f}", f"{r[8]:.4f}"])
    print(f"[csv]  per-query timeline written: {out_csv}\n")


def concurrency_profile(g, total_budget):
    """Build an event-driven timeline of in-flight requests."""
    events = []
    for r in g["requests"]:
        events.append((r["t_pickup"], +1, r["source"]))
        events.append((r["t_done"], -1, r["source"]))
    events.sort()

    main_in = spec_in = 0
    last_t = 0.0
    # accumulators
    time_main = defaultdict(float)        # bucket by main count
    time_total = defaultdict(float)       # bucket by total count
    time_at_cap = 0.0                     # total >= cap
    time_idle = 0.0                       # total == 0
    time_slack_no_spec = 0.0              # main < cap and spec == 0
    total_t = 0.0
    for t, d, src in events:
        dt = t - last_t
        if dt > 0:
            time_main[main_in] += dt
            time_total[main_in + spec_in] += dt
            total_t += dt
            if main_in + spec_in >= total_budget:
                time_at_cap += dt
            if main_in + spec_in == 0:
                time_idle += dt
            if main_in < total_budget and spec_in == 0:
                time_slack_no_spec += dt
        if src == "main":
            main_in += d
        else:
            spec_in += d
        last_t = t

    print(f"Concurrency profile (cap M = {total_budget}):")
    print(f"  total wall time observed: {total_t:7.2f}s")
    print(f"  fully idle:               {time_idle:7.2f}s "
          f"({100*time_idle/total_t:5.1f}%)")
    print(f"  saturated (in_flight=={total_budget}): "
          f"{time_at_cap:7.2f}s ({100*time_at_cap/total_t:5.1f}%)")
    print(f"  slack but no spec issued:  {time_slack_no_spec:7.2f}s "
          f"({100*time_slack_no_spec/total_t:5.1f}%)  <-- spec missed slot here")
    # weighted mean total in-flight
    mean_inflight = sum(c * dt for c, dt in time_total.items()) / total_t
    print(f"  mean in-flight: {mean_inflight:.2f}")
    print(f"  in-flight distribution:")
    for c in sorted(time_total):
        if time_total[c] / total_t < 0.005:
            continue
        print(f"     {c:>3}: {time_total[c]:6.2f}s  ({100*time_total[c]/total_t:5.1f}%)")
    print()
    return dict(total=total_t, idle=time_idle, at_cap=time_at_cap,
                slack=time_slack_no_spec, mean_inflight=mean_inflight)


def cross_query_overlap(g):
    """Look at how queries overlap on the timeline."""
    reqs = g["requests"]
    by_q = defaultdict(list)
    for r in reqs:
        by_q[r["query_id"]].append(r)
    spans = []
    for qid in sorted(by_q):
        rs = by_q[qid]
        spans.append((qid, min(r["t_enqueue"] for r in rs),
                      max(r["t_done"] for r in rs)))
    spans.sort(key=lambda x: x[1])
    print("Query lifespans (start..end):")
    for qid, s, e in spans:
        print(f"  q{qid:<2} [{s:6.2f} .. {e:6.2f}]  (len {e-s:5.2f})")
    print()

    # average pairwise overlap
    overlaps = 0
    total = 0
    for i, (q1, s1, e1) in enumerate(spans):
        for q2, s2, e2 in spans[i + 1:]:
            ov = max(0.0, min(e1, e2) - max(s1, s2))
            if ov > 0:
                overlaps += ov
                total += 1
    if total:
        print(f"  avg overlap among pairs that share time: {overlaps/total:.2f}s "
              f"({total} overlapping pairs)\n")


def slack_windows(g, total_budget, min_dur=0.05):
    """Print individual slack windows where in_flight < cap and no spec was issued.

    These are the moments where a smarter cap policy could have inserted more
    speculative work without violating the global budget.
    """
    events = []
    for r in g["requests"]:
        events.append((r["t_pickup"], +1, r["source"]))
        events.append((r["t_done"], -1, r["source"]))
    events.sort()

    main_in = spec_in = 0
    last_t = 0.0
    windows = []  # (start, end, slots_free, main_in, spec_in)
    cur_open = None
    for t, d, src in events:
        if t > last_t:
            slots_free = total_budget - (main_in + spec_in)
            if slots_free > 0 and main_in > 0 and spec_in == 0:
                if cur_open is None:
                    cur_open = (last_t, slots_free, main_in)
            else:
                if cur_open is not None:
                    s_t, slots, mi = cur_open
                    if t - s_t >= min_dur:
                        windows.append((s_t, t, slots, mi))
                    cur_open = None
        if src == "main":
            main_in += d
        else:
            spec_in += d
        last_t = t

    print(f"Slack windows >= {min_dur}s where main is active, no spec is in flight, "
          f"and free slots exist:")
    print(f"  {'#':>3} {'start':>7} {'end':>7} {'dur':>6} {'free':>5} {'main_in':>8}")
    total_slack = 0.0
    for i, (s, e, slots, mi) in enumerate(sorted(windows, key=lambda x: -(x[1] - x[0]))[:20]):
        print(f"  {i:>3} {s:7.2f} {e:7.2f} {e - s:6.2f} {slots:>5} {mi:>8}")
        total_slack += (e - s)
    if len(windows) > 20:
        print(f"  ... ({len(windows) - 20} more)")
    print(f"  total slack in {len(windows)} windows >= {min_dur}s: {total_slack:.2f}s")
    print()


def per_query_useful_spec_dist(g, per_query):
    """Per-query histogram of useful-spec ratio."""
    by_q = defaultdict(list)
    for r in g["requests"]:
        by_q[r["query_id"]].append(r)

    chosen_by_q = {}
    for qid, td in per_query.items():
        nodes = td.get("nodes", []) if isinstance(td, dict) else td
        chosen_by_q[qid] = {n["id"] for n in nodes if n.get("was_main_chosen")}

    buckets = [0] * 5  # 0-20%, 20-40%, ...
    for qid in by_q:
        rs = by_q[qid]
        n_spec = sum(1 for r in rs if r["source"] == "spec")
        chosen = chosen_by_q.get(qid, set())
        n_used = sum(1 for r in rs
                     if r["source"] == "spec" and r["ok"] and r["child_id"] in chosen)
        pct = 100.0 * n_used / n_spec if n_spec else 0.0
        b = min(4, int(pct // 20))
        buckets[b] += 1

    print("Useful-spec % distribution across queries:")
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    for label, n in zip(labels, buckets):
        bar = "#" * n
        print(f"  {label:>7}: {n:>3}  {bar}")
    print()


def thread_alignment(g):
    """For each pair of overlapping queries, see how often both are
    speculating at the same time vs alternating."""
    by_q = defaultdict(list)
    for r in g["requests"]:
        by_q[r["query_id"]].append(r)

    spans = {}
    for qid in by_q:
        rs = by_q[qid]
        spans[qid] = (min(r["t_enqueue"] for r in rs), max(r["t_done"] for r in rs))

    qids = sorted(spans)
    print("Co-running query pairs: shared time, fraction of shared time both spec / one spec / neither spec")
    print(f"  {'q1':>3} {'q2':>3} {'shared':>7} {'both_spec':>10} {'one_spec':>9} {'no_spec':>8}")
    pairs_processed = 0
    for i, q1 in enumerate(qids):
        for q2 in qids[i + 1:]:
            s = max(spans[q1][0], spans[q2][0])
            e = min(spans[q1][1], spans[q2][1])
            if e - s < 0.5:
                continue
            # build event list across both queries
            events = []
            for q in (q1, q2):
                for r in by_q[q]:
                    if r["source"] == "spec" and r["t_pickup"] < e and r["t_done"] > s:
                        events.append((max(s, r["t_pickup"]), +1, q))
                        events.append((min(e, r["t_done"]), -1, q))
            events.sort()
            counts = {q1: 0, q2: 0}
            both = one = neither = 0.0
            last = s
            for t, d, q in events:
                dt = t - last
                if dt > 0:
                    a = counts[q1] > 0
                    b = counts[q2] > 0
                    if a and b: both += dt
                    elif a or b: one += dt
                    else: neither += dt
                counts[q] += d
                last = t
            dt = e - last
            if dt > 0:
                a = counts[q1] > 0
                b = counts[q2] > 0
                if a and b: both += dt
                elif a or b: one += dt
                else: neither += dt
            tot = both + one + neither
            if tot < 0.5:
                continue
            pairs_processed += 1
            if pairs_processed <= 10:
                print(f"  q{q1:<2} q{q2:<2} {tot:7.2f} "
                      f"{100*both/tot:9.1f}% {100*one/tot:8.1f}% {100*neither/tot:7.1f}%")
    print()


def main():
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "exp_results/mt_trace/traces"
    g, per_query = load(trace_dir)

    print("=" * 78)
    print(f"Trace dir: {trace_dir}")
    print(f"  num_threads={g['num_threads']}  width={g['width']}  "
          f"speculative={g['speculative']}  total_budget={g['total_budget']}")
    print(f"  total requests: {len(g['requests'])}  queries: {len(per_query)}")
    print("=" * 78 + "\n")

    out_csv = os.path.join(trace_dir, "global_requests.csv")
    per_request_table(g, out_csv, head=20)
    rows = per_query_summary(g, per_query)
    write_query_csv(rows, os.path.join(trace_dir, "queries.csv"))
    concurrency_profile(g, g["total_budget"])
    cross_query_overlap(g)
    slack_windows(g, g["total_budget"], min_dur=0.05)
    per_query_useful_spec_dist(g, per_query)
    thread_alignment(g)


if __name__ == "__main__":
    main()
