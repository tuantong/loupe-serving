import math

from loupe.policy import average_over_orders, cyclic_orders, margin


def test_cyclic_orders_start_with_identity_and_are_permutations():
    orders = cyclic_orders(5, 4)
    assert orders[0] == [0, 1, 2, 3, 4]
    assert len(orders) == 4
    for o in orders:
        assert sorted(o) == [0, 1, 2, 3, 4]
    assert len({tuple(o) for o in orders}) == 4


def test_cyclic_orders_caps_k_at_n():
    assert cyclic_orders(2, 4) == [[0, 1], [1, 0]]


def test_average_maps_back_to_original_indices():
    orders = [[0, 1], [1, 0]]
    probs = [[0.9, 0.1], [0.3, 0.7]]
    avg = average_over_orders(probs, orders)
    assert math.isclose(avg[0], (0.9 + 0.7) / 2)
    assert math.isclose(avg[1], (0.1 + 0.3) / 2)


def test_margin_is_top_two_gap():
    assert math.isclose(margin([0.6, 0.3, 0.1]), 0.3)
    assert math.isclose(margin([0.5, 0.5]), 0.0)
    assert margin([1.0]) == 1.0
