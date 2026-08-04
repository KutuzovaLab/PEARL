def get_consecutive_subsets(lst):
    lst = sorted(lst)
    subsets = []
    current = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current.append(lst[i])
        else:
            if len(current) > 1:
                subsets.append(current)
            current = [lst[i]]

    if len(current) > 1:
        subsets.append(current)

    return subsets
