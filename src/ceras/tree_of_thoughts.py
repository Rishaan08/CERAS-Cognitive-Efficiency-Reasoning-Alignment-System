# tree_of_thoughts.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Callable, Tuple
import uuid
import json
import time
import heapq


@dataclass
class ThoughtNode:
    id: str
    parent_id: Optional[str]
    text: str
    role: Optional[str] = None                    # e.g., "root", "subtask", "suggestion", "expansion"
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)  # store child IDs
    created_at: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None       # optional: for similarity retrieval

    def to_dict(self):
        return asdict(self)


class TreeOfThoughts:
    def __init__(self, tree_id: Optional[str] = None):
        self.tree_id = tree_id or str(uuid.uuid4())
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, text: str, parent_id: Optional[str] = None, role: Optional[str] = None,
                 scores: Optional[Dict[str, float]] = None, metadata: Optional[Dict[str, Any]] = None,
                 embedding: Optional[List[float]] = None) -> ThoughtNode:
        nid = str(uuid.uuid4())
        node = ThoughtNode(
            id=nid,
            parent_id=parent_id,
            text=text,
            role=role,
            scores=scores or {},
            metadata=metadata or {},
            embedding=embedding,
        )
        self.nodes[nid] = node
        if parent_id:
            parent = self.nodes.get(parent_id)
            if parent:
                parent.children.append(nid)
        else:
            # If no parent provided and root not set, set as root.
            if self.root_id is None:
                self.root_id = nid
        return node

    def set_root(self, node_id: str):
        if node_id not in self.nodes:
            raise KeyError("Node id not in tree")
        self.root_id = node_id
        self.nodes[node_id].parent_id = None

    def get_node(self, node_id: str) -> ThoughtNode:
        return self.nodes[node_id]

    def traverse(self, node_id: Optional[str] = None, depth: int = 0):
        nid = node_id or self.root_id
        if nid is None:
            return
        node = self.nodes[nid]
        yield (depth, node)
        for c in node.children:
            yield from self.traverse(c, depth + 1)

    def to_dict(self):
        return {
            "tree_id": self.tree_id,
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(path: str) -> "TreeOfThoughts":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tree = TreeOfThoughts(tree_id=data.get("tree_id"))
        for nid, nd in data["nodes"].items():
            node = ThoughtNode(
                id=nd["id"],
                parent_id=nd["parent_id"],
                text=nd["text"],
                role=nd.get("role"),
                scores=nd.get("scores", {}),
                metadata=nd.get("metadata", {}),
                children=nd.get("children", []),
                created_at=nd.get("created_at", time.time()),
                embedding=nd.get("embedding"),
            )
            tree.nodes[nid] = node
        tree.root_id = data.get("root_id")
        return tree

    # Utility: return top-k nodes by a named score key
    def top_k_by_score(self, score_key: str, k: int = 5) -> List[ThoughtNode]:
        heap: List[Tuple[float, str]] = []
        for nid, node in self.nodes.items():
            val = node.scores.get(score_key)
            if val is not None:
                if len(heap) < k:
                    heapq.heappush(heap, (val, nid))
                else:
                    heapq.heappushpop(heap, (val, nid))
        result = sorted(heap, key=lambda x: x[0], reverse=True)
        return [self.nodes[nid] for _, nid in result]

    # Simple keyword search across nodes
    def search(self, keyword: str) -> List[ThoughtNode]:
        kw = keyword.lower()
        return [n for n in self.nodes.values() if kw in (n.text or "").lower()]

    # Simple path scoring: sum of a given score_key along path
    def best_path(self, score_key: str, max_depth: int = 6) -> Tuple[List[ThoughtNode], float]:
        if self.root_id is None:
            return [], 0.0

        best_path: List[str] = []
        best_score = float("-inf")

        def dfs(current_id: str, path: List[str], acc_score: float, depth: int):
            nonlocal best_path, best_score
            if depth > max_depth:
                return
            node = self.nodes[current_id]
            node_score = node.scores.get(score_key, 0.0)
            new_score = acc_score + node_score
            new_path = path + [current_id]
            # If leaf or depth limit, consider
            if not node.children or depth == max_depth:
                if new_score > best_score:
                    best_score = new_score
                    best_path = new_path
            for c in node.children:
                dfs(c, new_path, new_score, depth + 1)

        dfs(self.root_id, [], 0.0, 0)
        return ([self.nodes[nid] for nid in best_path], best_score)

    # Expand tree by calling an expansion function per node. The expand_fn receives (node_text, metadata) and returns a list of candidate dicts:
    # [{"text": "...", "role":"suggestion", "scores": {...}, "metadata": {...}}]
    def expand_with(self, node_id: str, expand_fn: Callable[[str, Dict[str, Any]], List[Dict[str, Any]]],
                    beam: int = 3):
        node = self.nodes[node_id]
        candidates = expand_fn(node.text, node.metadata) or []
        # naive beam: take top N by 'score' if provided else first N
        scored = []
        for cand in candidates:
            # if candidate has 'scores' and a 'combined' key, use that, else zero
            s = cand.get("scores", {})
            combined = s.get("combined_score") or s.get("confidence") or 0.0
            scored.append((combined, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, cand in scored[:beam]:
            self.add_node(
                text=cand.get("text", ""),
                parent_id=node_id,
                role=cand.get("role", "suggestion"),
                scores=cand.get("scores"),
                metadata=cand.get("metadata"),
                embedding=cand.get("embedding")
            )
