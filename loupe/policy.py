def cyclic_orders(n: int, k: int) -> list[list[int]]:
    k = min(k, n)
    shifts = [round(i * n / k) for i in range(k)]
    return [[(pos + shift) % n for pos in range(n)] for shift in shifts]


def average_over_orders(probs_per_order: list[list[float]], orders: list[list[int]]) -> list[float]:
    n = len(orders[0])
    total = [0.0] * n
    for probs, order in zip(probs_per_order, orders):
        for pos, index in enumerate(order):
            total[index] += probs[pos]
    return [t / len(orders) for t in total]


def margin(probs: list[float]) -> float:
    if len(probs) < 2:
        return 1.0
    top = sorted(probs, reverse=True)
    return top[0] - top[1]
