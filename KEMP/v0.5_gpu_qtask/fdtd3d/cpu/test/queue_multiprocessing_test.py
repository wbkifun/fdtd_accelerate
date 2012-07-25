#from Queue import Queue
#from threading import Thread
import atexit
import numpy as np
from multiprocessing import JoinableQueue, Process, Event
from time import sleep


def f(arr, i):
    sleep(0.1)
    arr += i*0.01
    print 'arr', arr


class QueueTask:
    def __init__(self):
        self.queue = JoinableQueue()
        self.event = Event()
        atexit.register( self.queue.join )

        process = Process(target=self.work)
        process.daemon = True
        process.start()


    def work(self):
        while True:
            func, args, wait_for = self.queue.get()

            for evt in wait_for: 
                evt.wait()
            func(*args)
            self.event.set()

            self.queue.task_done()


    def enqueue(self, func, args=[], wait_for=[]):
        self.event.clear()
        self.queue.put( (func, args, wait_for) )

        return self.event 



if __name__ == '__main__':
    arr = np.ones(10)
    qtask  = QueueTask() 
    for i in xrange(20):
        qtask.enqueue(f, [arr, i])
        '''
        if i == 10:
            evt = Event()
            evt.clear()
            qtask.enqueue(f, [i], [evt])
        else:
            qtask.enqueue(f, [i])
        '''
