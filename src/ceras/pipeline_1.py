# pipeline_1.py
# Improved pipeline runner that builds a richer Tree-of-Thoughts.
# Exports: main(PROMPT) which runs the pipeline and saves:
#  - tree_of_thoughts_example.json
#  - pipeline_debug.log
#
# No try/except blocks are included (per your preference).

from tree_of_thoughts import TreeOfThoughts
from inference import run_inference_pipeline, run_decomposer
from CAMRE_EDU import combined_reasoning_score
import json
import io
import contextlib
from typing import Any, List
import time

# -------------------- utilities --------------------
def safe_get_text(item: Any) -> str:
    """Return readable text from str/dict/list/tuple or join nested items."""
    if item is None:
        return ""
    if isinstance(item, str):
        return " ".join(item.split())
    if isinstance(item, (list, tuple)):
        parts = [safe_get_text(x) for x in item]
        return " ".join([p for p in parts if p])
    if isinstance(item, dict):
        # prioritize keys
        for k in ("text", "input", "output", "content", "message", "prompt", "title", "step", "orig"):
            if k in item and item[k] is not None:
                return safe_get_text(item[k])
        parts = []
        for v in item.values():
            if isinstance(v, (str, list, tuple, dict)):
                parts.append(safe_get_text(v))
        return " ".join([p for p in parts if p])
    return str(item)

def normalize_text(text: Any) -> str:
    return safe_get_text(text).strip()

def is_control_text(text: str) -> bool:
    """Detect trivial control tokens we should not add as substantive nodes."""
    if not isinstance(text, str):
        return True
    t = text.strip()
    if not t:
        return True
    if len(t) < 3:
        return True
    tl = t.lower()
    if tl in ("accepted", "rejected", "ok", "yes", "no", "true", "false"):
        return True
    # control phrases
    if tl.startswith(("subtasks", "debug", "warning", "verifier", "fallback", "note:", "error", "pipeline", "field:")):
        return True
    if "forwarding inference" in tl or "subtasks sufficient" in tl:
        return True
    if all(not ch.isalnum() for ch in tl):
        return True
    return False

def numeric_scores_from_reasoning(reasoning: Any):
    if reasoning is None:
        return None
    if isinstance(reasoning, (int, float)):
        return float(reasoning)
    if isinstance(reasoning, dict):
        for key in ("reasoning_score", "combined", "combined_score", "confidence", "score", "overall"):
            v = reasoning.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None

def ensure_numeric_scores(d: Any):
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}

# dedupe helper for candidate dicts (by text)
def dedupe_candidates_by_text(cands: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for c in cands:
        t = normalize_text(c.get("text", "")).lower()
        if not t:
            continue
        key = t[:500]
        if key in seen:
            # optionally merge metadata or keep the higher-scored one later
            continue
        seen.add(key)
        out.append(c)
    return out

# -------------------- pipeline output normalization --------------------
def pipeline_to_candidates(pipe_out: Any) -> List[dict]:
    """
    Normalize pipeline output (mixed shapes) into a list of candidate dicts:
      {"text","role","scores","metadata"}
    This is permissive: it will extract text from nested metadata fields so useful content
    stored in metadata will be converted into nodes.
    """
    candidates: List[dict] = []
    if pipe_out is None:
        return candidates

    if not isinstance(pipe_out, dict):
        if isinstance(pipe_out, str):
            text = normalize_text(pipe_out)
            candidates.append({"text": text, "role": "raw", "scores": {}, "metadata": {"orig": pipe_out}})
        elif isinstance(pipe_out, (list, tuple)):
            for it in pipe_out:
                text = normalize_text(it)
                candidates.append({"text": text, "role": "raw", "scores": {}, "metadata": {"orig": it}})
        return candidates

    # helper ingestion that extracts text from many shapes
    def ingest(item, role_hint):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for it in item:
                text = normalize_text(it)
                meta = it if isinstance(it, dict) else {"orig": it}
                scores = ensure_numeric_scores(it.get("scores") if isinstance(it, dict) else {})
                candidates.append({"text": text, "role": role_hint, "scores": scores, "metadata": meta})
        else:
            text = normalize_text(item)
            meta = item if isinstance(item, dict) else {"orig": item}
            scores = ensure_numeric_scores(item.get("scores") if isinstance(item, dict) else {})
            candidates.append({"text": text, "role": role_hint, "scores": scores, "metadata": meta})

    # prioritized fields
    ingest(pipe_out.get("final_subtasks"), "final_subtask")
    ingest(pipe_out.get("suggestions"), "suggestion")
    ingest(pipe_out.get("outputs"), "output")
    ingest(pipe_out.get("thoughts"), "thought")
    ingest(pipe_out.get("intermediate_steps"), "thought")

    # permissive ingestion of other top-level fields
    known = {"final_subtasks", "suggestions", "outputs", "thoughts", "intermediate_steps"}
    for k, v in pipe_out.items():
        if k in known:
            continue
        if isinstance(v, (str, list, tuple, dict)):
            ingest(v, role_hint=f"field:{k}")

    # if nothing found, fallback to any top-level text/content
    if not candidates:
        fallback = normalize_text(pipe_out.get("text") or pipe_out.get("content") or pipe_out)
        if fallback:
            candidates.append({"text": fallback, "role": "fallback", "scores": {}, "metadata": {"orig": pipe_out}})
    return candidates

# -------------------- decomposer helper --------------------
def ensure_final_subtasks(out: Any, prompt: str) -> (List[Any], str):
    """
    Ensure we return a list of final subtasks. If pipeline didn't provide them,
    run run_decomposer(prompt) and return its list (and captured debug).
    """
    if isinstance(out, dict) and "final_subtasks" in out:
        fs = out["final_subtasks"]
        if isinstance(fs, (list, tuple)) and any(normalize_text(x) for x in fs):
            return fs, ""
        if isinstance(fs, str) and normalize_text(fs):
            return [fs], ""
    # else run decomposer and capture its prints
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        decomposed = run_decomposer(prompt)
    debug = buf.getvalue()
    if isinstance(decomposed, dict):
        for key in ("final_subtasks", "subtasks", "steps", "decomposition"):
            if key in decomposed and isinstance(decomposed[key], (list, tuple)):
                return decomposed[key], debug
        return [decomposed], debug
    if isinstance(decomposed, (list, tuple)):
        return list(decomposed), debug
    return [decomposed], debug

# -------------------- expansion --------------------
def expand_using_pipeline(prompt_text: str, metadata=None) -> List[dict]:
    """
    Call run_inference_pipeline(prompt_text, auto_extend=False) and return normalized candidates.
    Also attach reasoning_snapshot to metadata and derive combined_score where available.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = run_inference_pipeline(prompt_text, auto_extend=False)
    debug = buf.getvalue()
    cands = pipeline_to_candidates(out)
    reasoning = combined_reasoning_score(out, prompt_text)
    combined_num = numeric_scores_from_reasoning(reasoning)
    for c in cands:
        c.setdefault("scores", {})
        if "combined_score" not in c["scores"] and combined_num is not None:
            c["scores"]["combined_score"] = combined_num
        if "combined_score" not in c["scores"]:
            meta_scores = c.get("metadata", {})
            if isinstance(meta_scores, dict) and isinstance(meta_scores.get("scores"), dict):
                vals = [v for v in meta_scores["scores"].values() if isinstance(v, (int, float))]
                if vals:
                    c["scores"]["combined_score"] = float(sum(vals) / len(vals))
        c["scores"].setdefault("combined_score", 0.0)
        c.setdefault("metadata", {})
        c["metadata"]["reasoning_snapshot"] = reasoning
        # attach raw pipeline debug for visibility
        c["metadata"]["__pipeline_debug"] = debug
    return cands

# -------------------- merge duplicate nodes (mark & reparent) --------------------
def merge_duplicate_nodes_marking(tree: TreeOfThoughts):
    """
    Merge duplicate nodes by normalized text. Keep the one with highest combined_score,
    reparent children of duplicates to the best node, and mark duplicates as merged.
    We don't delete nodes to avoid depending on a remove API.
    """
    text_to_best = {}
    for nid, node in tree.nodes.items():
        txt = normalize_text(node.text)
        if not txt:
            continue
        key = txt.lower()[:500]
        score = float(node.scores.get("combined_score", 0.0))
        if key not in text_to_best or score > text_to_best[key][0]:
            text_to_best[key] = (score, nid)

    # now reparent duplicates into best node
    for key, (best_score, best_id) in text_to_best.items():
        for nid, node in list(tree.nodes.items()):
            if nid == best_id:
                continue
            if normalize_text(node.text).lower()[:500] == key:
                # move children to best node (avoid duplicates)
                for child in list(node.children):
                    if child not in tree.nodes[best_id].children:
                        tree.nodes[best_id].children.append(child)
                        tree.nodes[child].parent_id = best_id
                # mark duplicate node as merged (clear text and role)
                node.metadata = node.metadata or {}
                node.metadata["merged_into"] = best_id
                node.metadata["was_score"] = node.scores.get("combined_score", 0.0)
                node.text = ""
                node.role = "merged"
                node.scores["combined_score"] = 0.0

# -------------------- main --------------------
def main(PROMPT: str):
    prompt = PROMPT

    # run full pipeline (capture printed debug to log)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = run_inference_pipeline(prompt, auto_extend=True)
    pipeline_debug = buf.getvalue()

    # ensure final subtasks (use decomposer if needed)
    final_subtasks, decomposer_debug = ensure_final_subtasks(out, prompt)

    # show formatted final_subtasks (for console)
    print("\n=== FORMATTED FINAL SUBTASKS ===")
    if final_subtasks:
        for i, s in enumerate(final_subtasks, start=1):
            print(f"{i}. {safe_get_text(s)}")
    else:
        print("No substantive final_subtasks produced.")

    # create the tree and root
    tree = TreeOfThoughts()
    root = tree.add_node(text=prompt, parent_id=None, role="root", metadata={"source": "user_prompt", "ts": time.time()})
    tree.set_root(root.id)

    # attach diagnostic to root
    reasoning_result = combined_reasoning_score(out, prompt)
    root_combined = numeric_scores_from_reasoning(reasoning_result)
    tree.nodes[root.id].metadata["reasoning_diagnostic"] = reasoning_result

    # build initial candidates from pipeline out (permissive)
    initial_candidates = pipeline_to_candidates(out)

    # if pipeline provided nothing meaningful, fall back to decomposed final_subtasks
    if not initial_candidates and final_subtasks:
        for s in final_subtasks:
            txt = normalize_text(s)
            if txt:
                initial_candidates.append({"text": txt, "role": "final_subtask", "scores": {}, "metadata": {"orig": s, "source": "decomposer"}})

    # enrich candidates: extract hidden text from metadata and normalize
    enriched_candidates: List[dict] = []
    for cand in initial_candidates:
        text = normalize_text(cand.get("text", ""))
        # if candidate text is control-like, try to extract from metadata fields
        if not text or is_control_text(text):
            meta = cand.get("metadata", {})
            # common metadata fields to inspect
            for key in ("orig", "final_subtasks", "final_subtask", "subtasks", "outputs", "output", "content", "message"):
                if key in meta:
                    extracted = normalize_text(meta[key])
                    if extracted and not is_control_text(extracted):
                        text = extracted
                        break
        # If still empty, but reasoning snapshot has a list, try to join them
        if (not text or is_control_text(text)) and isinstance(cand.get("metadata", {}).get("reasoning_snapshot"), dict):
            snap = cand["metadata"]["reasoning_snapshot"]
            for key in ("final_subtasks", "subtasks", "steps"):
                if key in snap and isinstance(snap[key], (list, tuple)):
                    parts = [normalize_text(x) for x in snap[key] if normalize_text(x) and not is_control_text(normalize_text(x))]
                    if parts:
                        text = " ; ".join(parts)
                        break
        cand["text"] = text
        enriched_candidates.append(cand)

    # dedupe enriched candidates by text
    enriched_candidates = dedupe_candidates_by_text(enriched_candidates)

    # compute per-candidate scores when missing or low (run a quick verifier pass)
    for cand in enriched_candidates:
        scores = cand.setdefault("scores", {})
        if ("combined_score" not in scores) or (not isinstance(scores.get("combined_score"), (int, float)) or scores.get("combined_score", 0.0) == 0.0):
            cand_text = normalize_text(cand.get("text", ""))
            if cand_text:
                sub_out = run_inference_pipeline(cand_text, auto_extend=False)
                sub_reasoning = combined_reasoning_score(sub_out, cand_text)
                sub_num = numeric_scores_from_reasoning(sub_reasoning)
                if isinstance(sub_num, (int, float)):
                    cand["scores"]["combined_score"] = float(sub_num)
                else:
                    cand["scores"]["combined_score"] = float(scores.get("combined_score", 0.0))
                cand.setdefault("metadata", {})
                cand["metadata"]["reasoning_snapshot_for_candidate"] = sub_reasoning

    # Add initial candidates to tree: make sure to add meaningful nodes
    for cand in enriched_candidates:
        txt = normalize_text(cand.get("text", ""))
        if not txt or is_control_text(txt):
            # skip if nothing usable
            continue
        scores = cand.get("scores", {})
        scores.setdefault("combined_score", root_combined if root_combined is not None else 0.0)
        # provenance
        meta = cand.get("metadata", {})
        meta.setdefault("source", "pipeline")
        meta.setdefault("added_at", time.time())
        tree.add_node(text=txt, parent_id=root.id, role=cand.get("role", "final_subtask"), scores=scores, metadata=meta)

    # If no children were added (very possible), make nodes from final_subtasks explicitly
    if not tree.nodes[root.id].children and final_subtasks:
        for s in final_subtasks:
            txt = normalize_text(s)
            if not txt or is_control_text(txt):
                continue
            node_meta = {"source": "decomposer", "orig": s, "added_at": time.time()}
            node_scores = {"combined_score": root_combined or 0.0}
            tree.add_node(text=txt, parent_id=root.id, role="final_subtask", scores=node_scores, metadata=node_meta)

    # Expand tree more aggressively: beam and depth control for richer tree
    beam = 4   # increased beam for more breadth
    depth = 3  # increased depth for richer chains
    for d in range(depth):
        frontier = []
        if d == 0:
            frontier = list(tree.nodes[root.id].children)
        else:
            # collect nodes at depth d under root
            def collect_at_depth(node_id, remaining):
                if remaining == 0:
                    frontier.append(node_id)
                    return
                for cid in tree.nodes[node_id].children:
                    collect_at_depth(cid, remaining - 1)
            for child_id in list(tree.nodes[root.id].children):
                collect_at_depth(child_id, d-1)

        for node_id in frontier:
            parent_node = tree.nodes[node_id]
            candidates = expand_using_pipeline(parent_node.text, parent_node.metadata)
            # dedupe & sort by combined_score
            candidates = dedupe_candidates_by_text(candidates)
            candidates_sorted = sorted(candidates, key=lambda x: x.get("scores", {}).get("combined_score", 0.0), reverse=True)[:beam]
            for cand in candidates_sorted:
                txt = normalize_text(cand.get("text", ""))
                if not txt or is_control_text(txt):
                    continue
                scores = cand.get("scores", {})
                scores.setdefault("combined_score", root_combined if root_combined is not None else 0.0)
                meta = cand.get("metadata", {})
                meta.setdefault("source", "expansion")
                meta.setdefault("parent_node", node_id)
                meta.setdefault("added_at", time.time())
                tree.add_node(text=txt, parent_id=node_id, role=cand.get("role", "thought"), scores=scores, metadata=meta)

    # Merge duplicate nodes (mark and reparent)
    merge_duplicate_nodes_marking(tree)

    # Print top nodes for quick console feedback
    top_nodes = tree.top_k_by_score("combined_score", k=12)
    top_nodes = [n for n in top_nodes if n.text and not is_control_text(n.text)]
    print("\nTop nodes by combined_score:")
    for n in top_nodes:
        print(f"- id={n.id[:8]} score={n.scores.get('combined_score')} role={n.role} text={n.text}")

    # Best path
    path_nodes, path_score = tree.best_path("combined_score", max_depth=10)
    path_nodes_clean = [n for n in path_nodes if n.text and not is_control_text(n.text)]
    print("\nBest path (by summed combined_score):", path_score)
    for n in path_nodes_clean:
        print(f" * [{n.role}] {n.text} (score={n.scores.get('combined_score')})")

    # Save tree and debug logs
    outpath = "tree_of_thoughts_example.json"
    tree.save_json(outpath)

    with open("pipeline_debug.log", "w", encoding="utf-8") as f:
        f.write("=== PIPELINE DEBUG ===\n\n")
        f.write(pipeline_debug)
        f.write("\n\n=== DECOMPOSER DEBUG ===\n\n")
        f.write(decomposer_debug)

    print(f"\nSaved tree to {outpath}")
    # print reasoning diagnostic
    print("\n=== REASONING DIAGNOSTIC ===")
    print(json.dumps(reasoning_result, indent=2))

    # return the tree for callers that want it
    return {
        "tree": tree,
        "out": out,
        "pipeline_debug": pipeline_debug,
        "decomposer_debug": decomposer_debug,
        "reasoning": reasoning_result
    }

