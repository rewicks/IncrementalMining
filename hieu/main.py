#!/usr/bin/env python3
from hashlib import algorithms_available
import os
import sys
import logging
import argparse
import numpy as np
from collections import Counter
import scipy.special

sys.path.append("..")
from utils import MySQL, Languages
from environment import Env, GetEnv, Dummy, isParallel
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


class Decider:
    def ChooseLink(self, state, probs):
        if len(state.link_queue) > 0:
            link = np.random.choice(state.link_queue, 1, p=probs)
            print("link", link)
            return link[0]
        else:
            return None

class RandomDecider(Decider):
    def CalcProbs(self, state):
        ret = np.empty([len(state.link_queue)])
        if len(state.link_queue) > 0:
            ret.fill(1./len(state.link_queue))
        else:
            return np.empty([])

class LinearDecider(Decider):
    def __init__(self):
        self.coefficients = np.array([5, 5, 5])
    def CalcProbs(self, state):
        print("self.languages", state.languages)
        features = state.get_features()
        #features = np.array(self.get_features())
        print("features", features)

        langCosts = features[:3]
        langCosts = scipy.special.softmax(langCosts)

        probs = np.empty([len(state.link_queue)])
        for linkIdx, link in enumerate(state.link_queue):
            costs = np.zeros([3])
            #print(link.language) 
            if link.language == state.languages[0]:
                costs[0] = langCosts[0]
            elif link.language == state.languages[1]:
                costs[1] = langCosts[1]
            else:
                costs[2] = langCosts[2]
            #print("costs", costs)
            linkCost = np.inner(self.coefficients, costs)
            #print("linkCost", linkCost)
            probs[linkIdx] = linkCost

        probs = scipy.special.softmax(probs)
        #print("probs", probs.shape, np.sum(probs))
        return probs

######################################################################################
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
    
def transition_on_link(env, state, link_to_crawl):
    crawled_child = env.crawl_child(link_to_crawl.id)
    if crawled_child.lang != link_to_crawl.language:
        logging.info(f"Crawled child was in language {crawled_child.lang} but wanted to crawl {link_to_crawl.language}")

    new_state = State2(languages=state.languages, link_queue_limit = state.link_queue_limit)

    for li in state.visited_links:
        new_state.visited_links.add(li)
    new_state.visited_links.add(crawled_child.url)

    for li in state.link_queue:
            new_state.add_link(li)

    for child_link in crawled_child.links:
        link = Link(link_text=child_link.text,
                    link_text_lang=child_link.textLang,
                    link_url=child_link.childNode.url,
                    link_url_id=child_link.childNode.urlId)
        new_state.add_link(link)

    for parallel_doc in state.parallel_documents:
        new_state.parallel_documents.append(parallel_doc)

    crawled_doc = MonolingualDocument(docid=crawled_child.urlId, langid=crawled_child.lang)
    aligned = False
    for monolingual_doc in state.monolingual_documents:
        if isParallel(env, crawled_doc, monolingual_doc):
            parallel_doc = ParallelDocument(len(new_state.parallel_documents))
            parallel_doc.add_document(crawled_doc.docid)
            parallel_doc.add_document(monolingual_doc.docid)
            new_state.parallel_documents.append(parallel_doc)
            aligned = True
        else:
            new_state.monolingual_documents.append(monolingual_doc)

    if not aligned:
        new_state.monolingual_documents.append(crawled_doc)
    return new_state

def get_reward(state, new_state):
    assert(new_state is not None)
    new_documents = len(new_state.parallel_documents) - len(state.parallel_documents)
    reward = new_documents * 100
    return reward


######################################################################################
def main(args):
    print("Starting")
    maxStep = 190000000000000000 #1000
    coefficients = np.array([5, 5, 5])

    sqlconn = MySQL(args.config_file)
    languages = Languages(sqlconn)
    langIds = [languages.GetLang(args.lang_pair.split('-')[0]), languages.GetLang(args.lang_pair.split('-')[1])] 
    # env = Dummy()
    env = GetEnv(args.config_file, languages, args.host_name)

    state = create_start_state_from_node(env.rootNode, langIds, args.linkQueueLimit)

    ep_reward = 0
    discount = 1.0
    gamma = 0.95
    docs = []
    if args.algorithm == 'random':
        decider = RandomDecider()
    elif args.algorithm == 'linear':
        decider = LinearDecider()
    else:
        decider = None

    for t in range(1, maxStep):
        probs = decider.CalcProbs(state)
        link = decider.ChooseLink(state, probs)
        if link is None: # No more links left to crawl
            break

        new_state = transition_on_link(env, state, link)
        if new_state is None:
            break

        # logging of stats for comparing models (not used for training)
        reward = get_reward(state, new_state)
        docs.append(str(len(new_state.parallel_documents)))
        ep_reward += discount * reward
        discount *= gamma
        ##############################################################


        state = new_state
        
    print("Finished")
    print(f"Reward: {ep_reward}")
    print(f"Documents: {','.join(docs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default="random")
    parser.add_argument('--config-file', default="../config.ini")
    parser.add_argument('--host-name', default="http://www.visitbritain.com/")
    parser.add_argument('--lang-pair', default="en-fr")
    parser.add_argument('--link-queue-limit', dest="linkQueueLimit", type=int, default=10000000, help="Maximum size of buckets of links")

    args = parser.parse_args()
    #print("cpu", args.cpu)
    #exit(1)
    main(args)