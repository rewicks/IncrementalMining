"""
This file will contain:
    - state class
    - things that occur within a state
"""

from collections import Counter

# Base class for all other states
class State():
    def __init__(self, link_queue_limit=1000):
        super(LargeState, self).__init__()

        # unless people protest I think a list of size k is
        # actually simpler than a queue
        self.link_queue = []
        self.link_queue_limit

        # these actually might be *too* large to keep, so maybe
        # just keep metadata from them
        self.monolingual_documents = []
        self.parallel_documents = []

    def add_children_links(self, links):
        self.link_queue += links
        while len(self.link_queue) > self.link_queue_limit:
            self.link_queue.pop(0)

    def get_summary_state(self):
        language_counter = Counter([x.language for x in self.monolingual_documents])
        return [y[1] for y in sorted(language_counter.most_common(), key = lambda x: x[0])]

############################################################
#                     STATE MEMBERS                        # 
############################################################

class Link:
    def __init__(self):
        pass

class MonolingualDocument:
    def __init__(self, language):
        self.language = language

class ParallelDocument:
    def __init__(self):
        pass

class RewardHistory:
    def __init__(self):
        pass

class DecisionHistory:
    def __init__(self):
        pass