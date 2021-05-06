import hashlib
import numpy as np
import os
import struct


def np_random(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        print("ERROR")
    seed = create_seed(seed)

    rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng, seed


def hash_seed(seed=None, max_bytes=8):
    if seed is None:
        seed = create_seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode("utf8")).digest()
    return _bigint_from_bytes(hash[:max_bytes])


def create_seed(a=None, max_bytes=8):
    if a is None:
        a = _bigint_from_bytes(os.urandom(max_bytes))
    elif isinstance(a, str):
        a = a.encode("utf8")
        a += hashlib.sha512(a).digest()
        a = _bigint_from_bytes(a[:max_bytes])
    elif isinstance(a, int):
        a = a % 2 ** (8 * max_bytes)
    else:
        print("error")

    return a


def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b"\0" * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum


def _int_list_from_bigint(bigint):
    if bigint < 0:
        print("error")
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints
