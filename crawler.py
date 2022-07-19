'''
This file will contain:
    - the SmartCrawler class

'''
import torch
import torch.nn as nn


from state import State



class SmartCrawler():
    def __init__(self, state):
        self.state = state
        self.decider = Decider()

    def get_features():
        # this would call all of the features on the state
        pass


class Decider(nn.Module):
    def __init__(self):
        super(Decider, self).__init__()




#####################################################
#                      ACTIONS                      #
#####################################################

def crawl_child(child, state):
    # takes as input a child link and the state
    # wgets/simulates wgetting the link
    # creates new state/returns updated state
    pass

def process_documents(state):
    # creates a new state
    # processes list of MonolingualDocuments
    # adds new ParallelDocuments to state
    # returns new state
    pass

def stop_processing(state):
    # I'm not sure this actually has to be a function?
    # sends signal to stop?
    # returns a null state?
    pass
