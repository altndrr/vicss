from itertools import accumulate


def group(iterable: list, elem_per_list: list[int]) -> list:
    """Group elements of a list into sublists.

    Args:
        iterable (list): List to group.
        elem_per_list (list[int]): Number of elements per sublist.
    """
    end = list(accumulate(elem_per_list))
    start = [0] + end[:-1]
    return [iterable[s:e] for s, e in zip(start, end)]
