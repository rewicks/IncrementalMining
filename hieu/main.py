#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.append("..")
from decider import Decider

class State():
    def get_features(self):
        pass

######################################################################################
def main():
    print("Starting")

    state = State()

    for t in range(1, maxStep):
        features = np.array(state.get_features())



    print("Finished")

main()
