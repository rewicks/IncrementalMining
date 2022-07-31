#!/usr/bin/env python3
from hashlib import algorithms_available
import os
import sys
import logging
import argparse
import numpy as np
from collections import Counter
import scipy.special
import tldextract
from matplotlib import pyplot as plt

from utils import MySQL, Languages, GetLanguages, allhostNames
from environment import Env, GetEnv, Dummy, isParallel
from trajectory import *

num_coeff = 6


######################################################################################
def infer(args, languages, langIds, envs):
    #print("args.coeffs", args.coeffs)
    assert(args.coeffs is not None)
    assert(len(args.coeffs) == num_coeff)

    for host_name, env in envs:
        resLinear = trajectory(env, langIds, args.linkQueueLimit, 'linear', args.maxStep, args.quiet, args.coeffs, args.gamma)
        resRandom = trajectory(env, langIds, args.linkQueueLimit, 'random', args.maxStep, args.quiet, args.coeffs, args.gamma)
        assert(len(resLinear[2]) == len(resRandom[2]))
        t = list(range(len(resLinear[2])))

        domain = tldextract.extract(host_name).domain
        print(domain, resLinear[1], resRandom[1])
        plt.figure()
        plt.plot(t, resLinear[2], label='Linear')
        plt.plot(t, resRandom[2], label='Random')
        plt.legend()
        plt.title(domain)
        #plt.show(block=False)

        plt.savefig(domain + '.png')
        


######################################################################################
def train(args, languages, langIds, env):
    # env = Dummy()

    def tryLinear(coeffs):
        ep_reward, auc, docs = trajectory(env[1], langIds, args.linkQueueLimit, 'linear', args.maxStep, args.quiet, coeffs, args.gamma)
        print("auc", auc, "ep_reward", ep_reward, "Params", coeffs)
        return -auc

    from skopt import gp_minimize, dummy_minimize
    range_bound = 100.0
    ranges = []
    for x in range(num_coeff):
        ranges.append((-range_bound, range_bound))

    if args.minimizeType == 'gp':
        res = gp_minimize(tryLinear,                  # the function to minimize
                    ranges,      # the bounds on each dimension of x
                    acq_func="EI",      # the acquisition function
                    n_calls=args.numIterations,         # the number of evaluations of f
                    n_initial_points=5,  # the number of random initialization points
                    noise="gaussian", # 0.1**4,       # the noise level (optional)
                    random_state=None)   # the random seed
    elif args.minimizeType == 'dummy':
        res = dummy_minimize(tryLinear,
                    ranges,
                    n_calls=args.numIterations,
                    random_state=None)

    print("x=", res.x)
    print("f(x^*)=%.4f" % (res.fun))

    # docsLinear = trajectory(env, langIds, args.linkQueueLimit, 'linear', args.maxStep, args.quiet, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 9999999999.0])

######################################################################################
def GetEnvs(hosts, languages, config_file):
    envs = []
    for host_name in hosts:
        print(host_name)
        env = GetEnv(config_file, languages, host_name)
        t = (host_name, env)
        envs.append(t)
    return envs
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', default = "train")
    #parser.add_argument('--algorithm', default="random")
    parser.add_argument('--quiet', action='store_true', default = False)
    parser.add_argument('--config-file', default="../config.ini")
    parser.add_argument('--train-hosts', nargs="+", default=["http://www.visitbritain.com/"])
    parser.add_argument('--test-hosts', nargs="+", default=allhostNames)
    parser.add_argument('--lang-pair', default="en-fr")
    parser.add_argument('--link-queue-limit', dest="linkQueueLimit", type=int, default=10000000, help="Maximum size of buckets of links")
    parser.add_argument('--max-step', dest="maxStep", type=int, default=10000000, help="Maximum number of steps in trajectory")
    parser.add_argument("--co-efficients", dest="coeffs", nargs=6, help="co-efficients. Only for infer", type=float, default=None)
    parser.add_argument("--num-iterations", dest="numIterations", type=int, default=10, help="Numer of training iterations")
    parser.add_argument('--gamma', type=float, default=0.999, help="Reward discount")                            
    parser.add_argument('--minimize-type', dest='minimizeType', default='gp', help="gp=Bayesian optimization using Gaussian Processes. dummy=Random search by uniform sampling")                            
                            
    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    if (args.quiet):
        logger = logging.getLogger()
        logger.disabled = True

    languages = GetLanguages(args.config_file)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 

    if args.do == "train":
        print("args.train_hosts", args.train_hosts)
        envs = GetEnvs(args.train_hosts, languages, args.config_file)
        env = envs[0]
        #docsRandom = trajectory(env, langIds, args.linkQueueLimit, 'random', args.maxStep, args.quiet)
        train(args, languages, langIds, env)
    elif args.do == "infer":
        envs = GetEnvs(args.test_hosts, languages, args.config_file)
        infer(args, languages, langIds, envs)
    else:
        abort("dunno")

if __name__ == "__main__":
    main()