import atexit
import numpy as np
from threading import Thread, Event
from Queue import Queue


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class QueueTask:
    def __init__(self):
        self.queue = Queue()
        atexit.register( self.queue.join )

        self.thread = Thread(target=self.work)
        self.thread.daemon = True
        self.thread.start()


    def work(self):
        while True:
            cl_func, cl_queue, gs, ls, args, wait_for, event = self.queue.get()

            for evt in wait_for: 
                evt.wait()
            cl_func(cl_queue, (gs,), (ls,), *args)
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
