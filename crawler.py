'''
This file will contain:
    - the SmartCrawler class

'''
from state import State


class SmartCrawler:
    def __init__():
        self.state = State()

    def get_features():
        # this would call all of the features on the state
        pass




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

