from itertools import islice


def minibatch(items, size):
    _items = iter(items)
    while True:
        batch = list(islice(_items, size))
        if len(batch) == 0:
            break
        yield batch
