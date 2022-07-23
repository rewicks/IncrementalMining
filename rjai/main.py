#!/usr/bin/env python3
from hashlib import algorithms_available
import os
import sys
import logging
import argparse
import numpy as np
from collections import Counter
import scipy.special

from utils import MySQL, Languages, GetLanguages
from environment import Env, GetEnv, Dummy, isParallel
from state import *
from decider import *
from matplotlib import pyplot as plt

def trajectory(env, langIds, linkQueueLimit, algorithm, maxStep, quiet, coeffs = None):
    state = create_start_state_from_node(env.rootNode, langIds, linkQueueLimit)

    ep_reward = 0
    discount = 1.0
    gamma = 0.95
    docs = []
    if algorithm == 'random':
        decider = RandomDecider()
    elif algorithm == 'linear':
        decider = LinearDecider(coeffs)
    else:
        decider = None

    for t in range(1, maxStep):
        probs = decider.CalcProbs(state)
        if probs is not None:
            link = decider.ChooseLink(state, probs)

            if link is None: # No more links left to crawl
                if not quiet:
                    print("Exhausted all links in the queue. Nothing left to do")
                break

            new_state = transition_on_link(env, state, link)
            if new_state is None:
                break
        else:
            break

        # logging of stats for comparing models (not used for training)
        reward = get_reward(state, new_state)
        docs.append(len(new_state.parallel_documents))
        ep_reward += discount * reward
        discount *= gamma
        ##############################################################

        state = new_state

    if not quiet:
        print(f"Reward: {ep_reward}")
        print(f"Documents: {','.join([str(x) for x in docs])}")

    crawl_histories = []
    for x in range(1):
        crawl_histories.append(docs)
    
    if not quiet:
        print("CRAWL HISTORIES")
        for x in crawl_histories:
            print(",".join([str(y) for y in x]))
        
        print("AUC of CRAWL HISTORIES")
        for x in crawl_histories:
            print(sum(x))

        print("Mean AUC for all crawl histories: ", np.mean([np.sum(x) for x in crawl_histories]))
    return docs

######################################################################################
def main(args):
    languages = GetLanguages(args.config_file)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    # env = Dummy()
    env = GetEnv(args.config_file, languages, args.host_name)
    #docsRandom = trajectory(env, langIds, args.linkQueueLimit, 'random', args.maxStep, args.quiet)

    def tryLinear(coeffs):
        ret = sum(trajectory(env, langIds, args.linkQueueLimit, 'linear', args.maxStep, args.quiet, coeffs))
        print("SUM:", ret, "Params:", coeffs)
        return -ret

    from skopt import gp_minimize
    range_bound = 100.0
    num_coeff = 6
    ranges = []
    for x in range(num_coeff):
        ranges.append((-range_bound, range_bound))
    res = tryLinear([-69.65934903738345, 100.0, -59.00014481560262, -100.0, -100.0, -27.433650440507023])
    print(res)

    # docsLinear = trajectory(env, langIds, args.linkQueueLimit, 'linear', args.maxStep, args.quiet, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 9999999999.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--algorithm', default="random")
    parser.add_argument('--quiet', action='store_true', default = False)
    parser.add_argument('--config-file', default="../config.ini")
    parser.add_argument('--host-name', default="http://www.gamersyde.com/")
    parser.add_argument('--lang-pair', default="en-fr")
    parser.add_argument('--link-queue-limit', dest="linkQueueLimit", type=int, default=10000000, help="Maximum size of buckets of links")
    parser.add_argument('--max-step', dest="maxStep", type=int, default=10000000, help="Maximum number of steps in trajectory")

    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    if (args.quiet):
        logger = logging.getLogger()
        logger.disabled = True

    main(args)
