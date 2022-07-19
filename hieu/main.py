#!/usr/bin/env python3
import os
import sys
sys.path.append("..")
from decider import Decider

######################################################################################
def main():
    print("Starting")

    counts = [23, 34, 57]

    decider = Decider()
    probs = decider.Decide(counts)
    print("probs", probs)

    print("Finished")

main()
