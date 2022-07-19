import argparse
import os, sys, logging
import torch

from utils import MySQL, Languages
from state import (State, 
                create_start_state_from_node,
                transition_on_lang_prob)
from environment import Env, GetEnv
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
    env = GetEnv(args.config_file, languages, args.host_name)

    start_state = create_start_state_from_node(env.rootNode, langIds)
    decider = ReinforceDecider(args, env, transition_on_lang_prob)

    ### some testing
    probs = torch.tensor([0.3, 0.7])
    action = torch.argmax(probs)
    new_state = transition_on_lang_prob(env, start_state, action)

    # Start Training
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


    args = parser.parse_args()
    main(args)