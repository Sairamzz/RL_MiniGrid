import heapq

class MaxPriorityQueue: # Helps pull out the highest priority state-action pairs for prioritized sweeping
    def __init__(self):
        self.heap = []

    # Push an item with a given priority onto the queue
    def push(self, priority, item):
        heapq.heappush(self.heap, (-priority, item))

    # Remove and return the item with the highest priority
    def pop(self):
        priority, item = heapq.heappop(self.heap)
        return -priority, item 

    # Just to check if the queue is empty
    def empty(self):
        return len(self.heap) == 0
