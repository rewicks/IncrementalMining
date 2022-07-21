#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from collections import Counter

sys.path.append("..")
from utils import MySQL, Languages
from environment import Env, GetEnv, Dummy
from state import Link, MonolingualDocument, ParallelDocument

class State2():
    def __init__(self, languages, link_queue_limit):

        # unless people protest I think a list of size k is
        # actually simpler than a queue
        self.link_queue = []
        self.link_queue_limit = link_queue_limit
        self.visited_links = set()

        # these actually might be *too* large to keep, so maybe
        # just keep metadata from them
        self.monolingual_documents = []
        self.parallel_documents = []

        self.languages = languages

    def add_link(self, link):
        if link.url not in self.visited_links:
            self.link_queue.append(link)
        if len(self.link_queue) > self.link_queue_limit:
            self.link_queue.pop(0)

    def get_features(self):
        # monolingual documents
        doc_targeted_language_counter = Counter([x.language for x in self.monolingual_documents if x.language in self.languages])
        doc_targeted_langs = [doc_targeted_language_counter[lang] for lang in self.languages]
        doc_non_targeted_language_counter = Counter([x.language for x in self.monolingual_documents if x.language not in self.languages])
        doc_non_targeted_langs = [sum(doc_non_targeted_language_counter)]

        # link counters
        link_targeted_language_counter = Counter([x.language for x in self.link_queue if x.language in self.languages])
        link_targeted_langs = [link_targeted_language_counter[lang] for lang in self.languages]
        link_non_targeted_language_counter = Counter([x.language for x in self.link_queue if x.language not in self.languages])
        link_non_targeted_langs = [sum(link_non_targeted_language_counter)]

        return doc_targeted_langs + doc_non_targeted_langs + link_targeted_langs + link_non_targeted_langs

    def CalcProbs(self, coefficients):
        features = np.array(self.get_features())
        print("features", features)
        probs = np.empty([len(self.link_queue)])
        for link in self.link_queue:
            #print(link.language) 
            pass

        print("probs", probs.shape)
        return probs

def create_start_state_from_node(root, languages, link_queue_limit):
    state = State2(languages, link_queue_limit=link_queue_limit)

    # update with the only crawled page
    root_document = MonolingualDocument(docid=root.urlId, langid=root.lang)
    state.monolingual_documents.append(root_document)

    # add all children
    for li in root.links:
        link = Link(link_text=li.text,
                    link_text_lang=li.textLang,
                    link_url=li.childNode.url,
                    link_url_id=li.childNode.urlId)
        state.add_link(link)

    return state
######################################################################################
def SafeDiv(a, b):
    if b == 0:
        return 10
    else:
        return a / b
    
######################################################################################
def main(args):
    print("Starting")
    maxStep = 2 #000000
    coefficients = [1, 2, 3, 4, 5, 6]

    sqlconn = MySQL(args.config_file)
    languages = Languages(sqlconn)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    # env = Dummy()
    env = GetEnv(args.config_file, languages, args.host_name)

    state = create_start_state_from_node(env.rootNode, langIds, args.linkQueueLimit)

    for t in range(1, maxStep):
        state.CalcProbs(coefficients)


    print("Finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default="../config.ini")
    parser.add_argument('--host-name', default="http://www.visitbritain.com/")
    parser.add_argument('--lang-pair', default="en-fr")
    parser.add_argument('--link-queue-limit', dest="linkQueueLimit", type=int, default=10000000, help="Maximum size of buckets of links")

    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    main(args)