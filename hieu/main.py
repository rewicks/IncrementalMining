#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.append("..")
from utils import MySQL, Languages
from environment import Env, GetEnv, Dummy

class State():
    def get_features(self):
        ret = np.zeros([3])
        return ret

######################################################################################
def main(args):
    print("Starting")
    maxStep = 1000000

    sqlconn = MySQL(args.config_file)
    languages = Languages(sqlconn)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    # env = Dummy()
    env = GetEnv(args.config_file, languages, args.host_name)

    state = State()

    for t in range(1, maxStep):
        features = np.array(state.get_features())
        print("features", features)


    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default="../config.ini")
    parser.add_argument('--host-name', default="http://www.visitbritain.com/")
    parser.add_argument('--lang-pair', default="en-fr")

    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    main(args)