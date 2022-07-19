import argparse
import os, sys, logging

from utils import MySQL, Languages
from state import State
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
    # langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    env = GetEnv(args.config_file, languages, args.host_name)
    # env = Env(sqlconn, args.host_name)

    # build desired state


    # build desired decider
    def env_step(action, state):
        return state # newstate

    decider = ReinforceDecider(args, env_step)

    # build crawler


    start_state = State()
    env.rootNode

    # Start Training
    decider.train(start_state)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default="config.ini")
    parser.add_argument('--host-name', default="http://www.visitbritain.com/")
    parser.add_argument('--lang-pair', default="en-fr")


    args = parser.parse_args()
    main(args)