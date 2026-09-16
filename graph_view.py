"""Static stream-graph and node metadata payloads."""

from inspection import (
    ABLATABLE_NODES,
    BRANCHES,
    DEEMBEDDABLE_NODES,
    DEFAULT_NODE_KEY,
    EMBEDDING_COMPONENTS,
    SPINE_LINKS,
    SPINE_NODES,
    STREAM_NODES,
    TRACE_ORDER,
    VOCAB_CONTRIBUTABLE_NODES,
    node_spec,
)


def graph_payload() -> dict:
    """Return the declarative wiring of the three-block model."""
    nodes = [
        {
            "key": spec.key,
            "label": spec.label,
            "kind": spec.kind,
            "family": spec.family,
            "explanation": spec.explanation,
            "normalized": spec.normalized,
            "feature_axis": spec.feature_axis,
            "deembeddable": spec.key in DEEMBEDDABLE_NODES,
            "vocab_contributable": spec.key in VOCAB_CONTRIBUTABLE_NODES,
            "block_index": spec.block_index,
            "stage": spec.stage,
            "width_source": spec.width_source,
        }
        for spec in STREAM_NODES
    ]
    trace = list(TRACE_ORDER)
    by_key = {item["key"]: item for item in nodes}
    for index, key in enumerate(trace):
        by_key[key]["trace_index"] = index
        by_key[key]["trace_count"] = len(trace)
        by_key[key]["prev_key"] = trace[index - 1] if index > 0 else None
        by_key[key]["next_key"] = trace[index + 1] if index < len(trace) - 1 else None
    return {
        "nodes": nodes,
        "spine": list(SPINE_NODES),
        "spine_links": list(SPINE_LINKS),
        "branches": [
            {
                "key": branch.key,
                "label": branch.label,
                "reads": branch.reads,
                "adds_before": branch.adds_before,
                "path": list(branch.path),
                "observables": list(branch.observables),
                "kind": branch.kind,
                "block_index": branch.block_index,
                "side": branch.side,
            }
            for branch in BRANCHES
        ],
        "components": list(EMBEDDING_COMPONENTS),
        "trace": trace,
        "default_node": DEFAULT_NODE_KEY,
    }


def node_info_payload(key: str) -> dict:
    """Describe one node and its position in the trace."""
    spec = node_spec(key)
    trace = list(TRACE_ORDER)
    index = trace.index(key)
    return {
        "key": spec.key,
        "label": spec.label,
        "kind": spec.kind,
        "family": spec.family,
        "explanation": spec.explanation,
        "normalized": spec.normalized,
        "feature_axis": spec.feature_axis,
        "deembeddable": key in DEEMBEDDABLE_NODES,
        "vocab_contributable": key in VOCAB_CONTRIBUTABLE_NODES,
        "block_index": spec.block_index,
        "stage": spec.stage,
        "width_source": spec.width_source,
        "trace_index": index,
        "trace_count": len(trace),
        "prev_key": trace[index - 1] if index > 0 else None,
        "next_key": trace[index + 1] if index < len(trace) - 1 else None,
    }


def options_payload() -> dict:
    """Return the static stream graph and available activation nodes."""
    return {
        "graph": graph_payload(),
        "locations": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "family": spec.family,
            }
            for spec in STREAM_NODES
        ],
        "ablation_nodes": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "family": spec.family,
            }
            for key in ABLATABLE_NODES
            for spec in (node_spec(key),)
        ],
    }


__all__ = ["graph_payload", "node_info_payload", "options_payload"]
