import sys
import time


class Timer(object):
    def __init__(self, text = None):
        if text != None:
            print(f'{text}: ', end='', flush=True), 
        self.start_time = time.perf_counter()

    def stop(self):
        self.elapsed = time.perf_counter() - self.start_time
        print(f'Time: {self.elapsed:.3f} seconds.')

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.stop()


class Profiler(object):
    def __init__(self, text = None):
        if text != None:
            print ('%s:' % text), 
        import cProfile
        self.pr = cProfile.Profile()
        self.pr.enable()
    def stop(self):
        self.pr.disable()
        import pstats
        ps = pstats.Stats(self.pr)
        ps.sort_stats('cumtime').print_stats(50)
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()
