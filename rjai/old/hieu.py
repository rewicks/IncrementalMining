#!/usr/bin/env python3
import os
import sys
import numpy as np
import argparse
from common import MySQL, GetLanguages, Languages, Timer
from helpers import GetVistedSiblings, GetMatchedSiblings, GetNodeMatched, GetEnvs, GetEnv, Env

######################################################################################
def main():
    print("Starting")

    configPath = 'config.ini'
    url = "http://www.buchmann.ch/"


    #sqlconn = MySQL(configPath)
    #env = Env(sqlconn, url)

    languages = GetLanguages(configPath)

    #GetEnv

    print("Finished")
    
######################################################################################
main()
