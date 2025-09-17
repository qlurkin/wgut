from time import perf_counter

__start = 0.0
__started = False


def start():
    global __started
    global __start
    if not __started:
        __start = perf_counter()
        __started = True


def stop() -> float:
    global __started
    if __started:
        T = perf_counter() - __start
        __started = False
        return T
    return 0.0
