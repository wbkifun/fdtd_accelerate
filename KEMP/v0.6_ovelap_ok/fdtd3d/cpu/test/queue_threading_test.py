import atexit
import numpy as np
from Queue import Queue
from threading import Thread, Event
from time import sleep


def f(arr, i):
    sleep(0.1)
    arr += i*0.01
    print 'arr', arr


class QueueTask:
    def __init__(self):
        self.queue = Queue()
        atexit.register( self.queue.join )

        self.thread = Thread(target=self.work)
        self.thread.daemon = True
        self.thread.start()


    def work(self):
        while True:
            f, args, wait_for, event = self.queue.get()

            for evt in wait_for:
                evt.wait()
            f(*args)
            event.set()

            self.queue.task_done()


    def enqueue(self, func, args=[], wait_for=[]):
        event = Event()
        event.clear()
        self.queue.put( (func, args, wait_for, event) )

        return event 



if __name__ == '__main__':
    arr = np.ones(10)
    qtask  = QueueTask() 
    for i in xrange(20):
        qtask.enqueue(f, [arr, i])




"""

def worker(q):
    while True:
        item = q.get()
        print(item)
        q.task_done()

q = JoinableQueue()
for i in range(1):
    p = Process(target=worker, args=[q])
    p.daemon = True
    p.start()

for item in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
    q.put(item)

q.join()
"""
