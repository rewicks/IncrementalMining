#!/usr/bin/env python3
import argparse
import os, sys, logging
import torch
import random

from utils import MySQL, Languages
from state import (State, 
                create_start_state_from_node,
                transition_on_lang_prob)
from environment import Env, GetEnv, Dummy
from reinforce import ReinforceDecider

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("SmartCrawler")


def main(args):
    sqlconn = MySQL(args.config_file)
    languages = Languages(sqlconn)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    # env = Dummy()
    env = GetEnv(args.config_file, languages, args.host_name)

    start_state = create_start_state_from_node(env.rootNode, langIds)
    decider = ReinforceDecider(args, env, transition_on_lang_prob, args.cpu, args.gamma, args.learningRate)

    ### some testing
    state = start_state
    # for y in range(5):
    #     state = start_state
    #     print('do-over')
    #     for x in range(10):
    #         features = state.get_features()#  [3,4]
    #         action = random.choice([0,1])
    #         if action == 0 and features[3] > 0:
    #             state = transition_on_lang_prob(env, state, action)
    #         elif action == 1 and features[4] > 0:
    #             state = transition_on_lang_prob(env, state, action)
    #         elif action == 0 and features[3] == 0:
    #             if features[4] > 0:
    #                 state = transition_on_lang_prob(env, state, 1)
    #         elif action == 1 and features[4] == 0:
    #             if features[3] > 0:
    #                 state = transition_on_lang_prob(env, state, 0)
    #         else:
    #             state = transition_on_lang_prob(env, state, 0)

    # probs = torch.tensor([0.3, 0.7])
    # action = torch.argmax(probs)
    # new_state = transition_on_lang_prob(env, start_state, action)

    # # Start Training
    decider.train(start_state)


    # load model 
    # model = load_model("beep boop")

    # training loop
    # while true:
        # decision = model.decide(current_state.get_features()) --> .get_features() will return number of monolingual documents in each language
        # new_state = state.update(decision)
        # reward = model.reinforce(current_state, new_state)
        # current_state = new_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default="config.ini")
    parser.add_argument('--host-name', default="http://www.visitbritain.com/")
    parser.add_argument('--lang-pair', default="en-fr")
    parser.add_argument('--cpu', dest="cpu", action='store_true')
    parser.add_argument('--gamma', type=float, default=0.999, help="Reward discount")
    parser.add_argument('--learning-rate', dest="learningRate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    main(args)