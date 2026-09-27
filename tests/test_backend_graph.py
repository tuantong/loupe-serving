import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and the model weights")]

MESSAGES = [
    [{"role": "user", "content": "Invoice 17 is due on 31 January. Is it overdue on 2 February? Answer A for yes, B for no."}],
    [{"role": "user", "content": "A lot of 12 units has 3 faulty ones. Two are drawn. Is a faulty draw more likely than not? A yes, B no. " * 8}],
]


def test_graph_replay_matches_the_plain_forward_pass():
    from loupe.backend_graph import GraphBackend
    from loupe.backend_hf import HFBackend

    graph = GraphBackend(step=32, max_graph_tokens=512)
    slices = [[graph.encode("A")[0], graph.encode("B")[0]]] * len(MESSAGES)
    fast = graph.slice_logits(MESSAGES, slices)
    plain = HFBackend.slice_logits_from_ids(graph, [graph._prompt_ids(m, False) for m in MESSAGES], slices)
    for a, b in zip(fast, plain):
        assert max(abs(x - y) for x, y in zip(a, b)) < 0.25
        assert a.index(max(a)) == b.index(max(b))
