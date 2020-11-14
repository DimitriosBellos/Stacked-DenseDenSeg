#from __future__ import absolute_import
import time


class Timer(object):
    def __init__(self):
        self.last_time = time.time()

    def get_value(self):
        if self.last_time:
            now = time.time()
            duration = now - self.last_time
            #self.last_time = now
            return duration
        else:
            self.last_time = time.time()
            return
