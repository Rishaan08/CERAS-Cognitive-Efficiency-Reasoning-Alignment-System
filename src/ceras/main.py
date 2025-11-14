# pipeline_output_generator.py
"""
Run pipeline_1.main(PROMPT), load and summarize results, save PNGs of the tree,
and expose `pipeline_1_output` for later reuse.

Usage:
    python pipeline_output_generator.py
"""

from pipeline_1 import main as run_full_pipeline
from tree_of_thoughts import TreeOfThoughts
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List

# ---------- Config ----------
TREE_JSON_PATH = "tree_of_thoughts_example.json"
TREE_IMG_FULL = "tree_of_thoughts_example.png"
TREE_IMG_SUB = "tree_of_thoughts_substantive.png"
PIPELINE_DEBUG_LOG = "pipeline_debug.log"

# Edit PROMPT as needed
PROMPT = "Teach me Agentic AI, agents and strands AWS from scratch"

# ---------- Helpers ----------
def normalize_text(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())

def is_control_text(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = normalize_text(text)
    if not t:
        return True
    if len(t) < 3:
        return True
    tl = t.lower()
    if tl in ("accepted", "rejected", "ok", "yes", "no"):
        return True
    if tl.startswith(("subtasks", "debug", "verifier", "fallback", "note:", "error", "pipeline", "field:")):
        return True
    if "forwarding inference" in tl or "subtasks sufficient" in tl:
        return True
    if all(not ch.isalnum() for ch in tl):
        return True
    return False

def role_is_substantive(role: str) -> bool:
    if not role:
        return False
    rl = role.lower()
    if rl in ("final_subtask", "suggestion", "output", "thought", "raw"):
        return True
    if rl.startswith("field:"):
        return False
    return True

def dedupe_texts(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        key = normalize_text(t).lower()[:300]
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out

def summarize_node(node):
    return {
        "id": node.id,
        "role": node.role,
        "text": node.text,
        "scores": node.scores,
        "num_children": len(node.children),
        "metadata_keys": list(node.metadata.keys()) if node.metadata else []
    }

# ---------- Run pipeline ----------
print("Running pipeline...")
result = run_full_pipeline(PROMPT)

# result should be a dict containing 'tree' according to pipeline_1.main
if isinstance(result, dict) and result.get("tree") is not None:
    tree = result["tree"]
else:
    # fallback: load from disk
    tree = TreeOfThoughts.load_json(TREE_JSON_PATH)

# ---------- Extract formatted final subtasks ----------
final_subtasks = []
# prefer explicit final_subtask nodes
for n in tree.nodes.values():
    if (n.role or "").lower() == "final_subtask":
        txt = normalize_text(n.text)
        if txt and not is_control_text(txt):
            final_subtasks.append(txt)

# fallback to suggestions
if not final_subtasks:
    for n in tree.nodes.values():
        if (n.role or "").lower() == "suggestion":
            txt = normalize_text(n.text)
            if txt and not is_control_text(txt):
                final_subtasks.append(txt)

# final fallback to any substantive node text
if not final_subtasks:
    for n in tree.nodes.values():
        txt = normalize_text(n.text)
        if txt and not is_control_text(txt):
            final_subtasks.append(txt)

final_subtasks = dedupe_texts(final_subtasks)

# ---------- Compute top nodes (filtered + deduped) ----------
top_nodes_objects = tree.top_k_by_score("combined_score", k=40)
top_nodes_filtered = []
seen = set()
for n in top_nodes_objects:
    txt = normalize_text(n.text)
    if not txt or is_control_text(txt):
        continue
    if not role_is_substantive(n.role):
        continue
    key = txt.lower()[:300]
    if key in seen:
        continue
    seen.add(key)
    top_nodes_filtered.append(n)
top_nodes = [summarize_node(n) for n in top_nodes_filtered]

# ---------- Best path (cleaned) ----------
path_nodes, path_score = tree.best_path("combined_score", max_depth=12)
path_nodes_clean = [n for n in path_nodes if normalize_text(n.text) and not is_control_text(n.text)]
best_path = [summarize_node(n) for n in path_nodes_clean]

# ---------- reasoning diagnostic ----------
reasoning_diagnostic = None
if tree.root_id:
    reasoning_diagnostic = tree.nodes[tree.root_id].metadata.get("reasoning_diagnostic")

# ---------- load pipeline debug if available ----------
pipeline_debug = None
if os.path.exists(PIPELINE_DEBUG_LOG):
    with open(PIPELINE_DEBUG_LOG, "r", encoding="utf-8") as fh:
        pipeline_debug = fh.read()

# ---------- Drawing helpers ----------
def draw_tree(t: TreeOfThoughts, outpath: str, filter_nodes: List = None, title: str = None):
    G = nx.DiGraph()
    labels = {}
    node_colors = []
    node_sizes = []
    nodes_to_draw = set()

    if filter_nodes is None:
        nodes_to_draw = set(t.nodes.keys())
    else:
        nodes_to_draw = set(n.id for n in filter_nodes)
        # include parents for context
        for n in filter_nodes:
            pid = n.parent_id
            while pid:
                nodes_to_draw.add(pid)
                pid = t.nodes.get(pid).parent_id if t.nodes.get(pid) else None

    for nid in nodes_to_draw:
        node = t.nodes[nid]
        display = normalize_text(node.text) or (node.role or nid[:8])
        if len(display) > 90:
            display = display[:87] + "..."
        labels[nid] = f"{display}\n({node.role})"
        combined = node.scores.get("combined_score") or node.scores.get("combined") or 0.0
        size = 300 + (max(0.0, float(combined)) ** 1.4) * 1000
        node_sizes.append(size)
        r = (node.role or "").lower()
        if r == "final_subtask":
            node_colors.append("#8dd3c7")
        elif r == "suggestion":
            node_colors.append("#ffffb3")
        elif r.startswith("field:"):
            node_colors.append("#bebada")
        else:
            node_colors.append("#fb8072")
        G.add_node(nid)

    for nid in nodes_to_draw:
        node = t.nodes[nid]
        for c in node.children:
            if c in nodes_to_draw:
                G.add_edge(nid, c)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, k=0.8, iterations=100)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=10)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ---------- Draw images ----------
draw_tree(tree, TREE_IMG_FULL, filter_nodes=None, title="Full Tree")

substantive_nodes = [n for n in tree.nodes.values() if normalize_text(n.text) and not is_control_text(n.text) and role_is_substantive(n.role)]
if not substantive_nodes and tree.root_id:
    substantive_nodes = [tree.nodes[cid] for cid in tree.nodes[tree.root_id].children]
draw_tree(tree, TREE_IMG_SUB, filter_nodes=substantive_nodes, title="Substantive Nodes (focused)")

# ---------- pipeline_1_output ----------
pipeline_1_output = {
    "tree_json_path": os.path.abspath(TREE_JSON_PATH),
    "tree_image_full": os.path.abspath(TREE_IMG_FULL),
    "tree_image_substantive": os.path.abspath(TREE_IMG_SUB),
    "formatted_final_subtasks": final_subtasks,
    "top_nodes": top_nodes,
    "best_path": {"nodes": best_path, "score": path_score},
    "reasoning_diagnostic": reasoning_diagnostic,
    "pipeline_debug_log": pipeline_debug,
    "tree_object": tree
}

# ---------- Print summary ----------
print("\n=== PIPELINE 1 SUMMARY (CLEAN) ===")
print("Saved tree JSON:", pipeline_1_output["tree_json_path"])
print("Saved tree image (full):", pipeline_1_output["tree_image_full"])
print("Saved tree image (substantive):", pipeline_1_output["tree_image_substantive"])
print("\nFormatted final subtasks:")
if pipeline_1_output["formatted_final_subtasks"]:
    for i, s in enumerate(pipeline_1_output["formatted_final_subtasks"], start=1):
        print(f" {i}. {s}")
else:
    print(" (no substantive final_subtasks found)")

print("\nTop nodes (cleaned):")
for n in pipeline_1_output["top_nodes"][:8]:
    print(f" - id={n['id'][:8]} score={n['scores'].get('combined_score')} role={n['role']} text={n['text'][:200]}")

print("\nBest path score:", pipeline_1_output["best_path"]["score"])
for n in pipeline_1_output["best_path"]["nodes"]:
    print("  *", n["role"], "-", n["text"][:200])
