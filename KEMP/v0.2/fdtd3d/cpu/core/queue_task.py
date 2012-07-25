import atexit
import numpy as np
from threading import Thread, Event
from Queue import Queue


class QueueTask(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.queue = Queue()
        atexit.register( self.queue.join )

        self.start()


    def run(self):
        while True:
            (lambda:None)()
            func, args, wait_for, event = self.queue.get()

            for evt in wait_for: 
                evt.wait()
            func(*args)
            event.set()

            self.queue.task_done()


    def enqueue(self, func, args=[], wait_for=[]):
        event = Event()
        event.clear()
        self.queue.put( (func, args, wait_for, event) )

        return event 


    def enqueue_barrier(self):
        evt = self.enqueue(lambda:None)
        evt.wait()
