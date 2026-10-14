# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Packing and verification contracts for the PTD host core."""

import numpy as np
import pytest

from torq.inference.ptd import PTDTree, PTDPool, greedy_verify, pack_round


def _two_node_paths():
    """candidate: 10 -> 11 ; draft: 20 -> 21, 20 -> 22."""
    cand = PTDTree(width=4, max_depth=4)
    c10 = cand.add_child(cand.root, 10)
    c11 = cand.add_child(c10, 11)
    draft = PTDTree(width=4, max_depth=4)
    d20 = draft.add_child(draft.root, 20)
    d21 = draft.add_child(d20, 21)
    d22 = draft.add_child(d20, 22)
    return cand, [c10, c11], draft, [d20, d21, d22]


def test_pack_round_layout_positions_and_vis():
    cand, cand_nodes, draft, draft_nodes = _two_node_paths()
    L = 5
    p = pack_round(L, anchor_token=99, candidate_nodes=cand_nodes,
                  draft_nodes=draft_nodes, rows=8, pad_token=0)
    # slot 0 = anchor; candidates first (BFS), then drafts
    assert p.tokens[0] == 99
    assert list(p.tokens[1:6]) == [10, 11, 20, 21, 22]
    assert p.positions[0] == L
    # sibling isolation: 21 and 22 share a position; 10/20 are first level
    assert p.positions[1] == L + 1   # cand depth 1
    assert p.positions[2] == L + 2   # cand depth 2
    assert p.positions[3] == L + 1   # draft depth 1
    assert p.positions[4] == L + 2   # draft depth 2 (21)
    assert p.positions[5] == L + 2   # draft depth 2 (22, sibling of 21)
    # unused slots are pad @ L
    assert list(p.tokens[6:]) == [0, 0]
    assert list(p.positions[6:]) == [L, L]

    assert p.vis.dtype == np.float32
    # vis (row, scratch-slot-1): row 2 (11) sees slot 1 (10) + itself
    assert p.vis[2, 0] == 1 and p.vis[2, 1] == 1
    # row 2 does not see draft rows (cross-tree isolation) or the pad rows
    assert (p.vis[2, 2:] == 0).all()
    # row 1 (10, first level) sees only itself
    assert p.vis[1, 0] == 1 and (p.vis[1, 1:] == 0).all()
    # row 4 (21) sees slot 3 (20) + itself, not sibling 22 (slot 5)
    assert p.vis[4, 2] == 1 and p.vis[4, 3] == 1
    assert p.vis[4, 4] == 0
    # row 5 (22) sees slot 3 (20) + itself, not sibling 21 (slot 4)
    assert p.vis[5, 2] == 1 and p.vis[5, 4] == 1 and p.vis[5, 3] == 0
    # anchor row sees no scratch rows
    assert (p.vis[0] == 0).all()
    # pad rows have zero visibility
    assert (p.vis[6] == 0).all() and (p.vis[7] == 0).all()
    assert p.n_candidates == 2 and p.n_drafts == 3
    assert p.candidate_slots[cand.root.children[10]] == 1
    assert p.draft_slots[draft.root.children[20].children[22]] == 5
    bounded = pack_round(L, 99, cand_nodes, draft_nodes, rows=4, pad_token=0)
    assert list(bounded.tokens) == [99, 10, 11, 20]
    assert bounded.n_candidates == 2 and bounded.n_drafts == 1
    pool = PTDPool(max_entry_nodes=1, max_entries=1)
    pool.merge_node(cand_nodes[0])
    assert pool.retrieve(10).node_count == 1
    pool.merge_node(draft_nodes[0])
    assert pool.retrieve(10) is None
    assert pool.retrieve(20).node_count == 1


def test_verification_respects_budget_and_stop_order():
    tree = PTDTree()
    first = tree.add_first_level(10)
    second = tree.add_child(first, 11)
    for budget, expected, stopped in [(0, [10], False), (1, [10, 11], False), (2, [10, 11, 12], True)]:
        seen = []

        def stop(token, committed):
            assert list(committed) == [5] + seen
            seen.append(token)
            return token == 12

        result = greedy_verify(np.array([10, 11, 12]), [first, second],
                               {first: 1, second: 2}, [5], stop, max_accepted=budget)
        assert result.emitted == expected
        assert result.accepted == [first, second][:budget]
        assert result.bonus == expected[-1]
        assert result.stopped is stopped
        assert seen == expected
    with pytest.raises(ValueError, match="max_accepted"):
        greedy_verify(np.array([10]), [], {}, [], stop, max_accepted=-1)
