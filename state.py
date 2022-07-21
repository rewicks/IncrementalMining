"""
This file will contain:
    - state class
    - things that occur within a state
"""
import torch
import environment
import logging
import os, sys

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("SmartCrawler")

from collections import Counter
import numpy as np

# Base class for all other states
class State():
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

    def isOption(self, language):
        language_counts = Counter([x.language for x in self.link_queue if x.language in self.languages])
        non_language_counts = Counter([x.language for x in self.link_queue if x.language not in self.languages])
        if language in language_counts:
            if language_counts[language] > 0:
                return True
            return False
        if sum(non_language_counts) > 0:
            return True
        return False

    def __str__(self):
        out = f"LINKS:\n"
        for li in self.link_queue:
            out += f'\t{li.url};{li.language}\n'

        return out


    
def transition_on_lang_prob(env, state, decision):
    # logging.info(f"Current state:{state}")
    # langid = torch.argmax(decision, dim=0)
    if decision < len(state.languages):
        langid = state.languages[int(decision)]
    else:
        langid = None

    link_to_crawl = None
    for li in state.link_queue:
        if li.language == langid:
            link_to_crawl = li
            break
        if langid is None and li.language not in state.languages:
            link_to_crawl = li
            break
    
    if link_to_crawl is None:
        logging.info(f"Crawler decided to crawl langid={langid} but no documents in this language remained in link queue.")
        return None
    
    crawled_child = env.crawl_child(link_to_crawl.id)
    if crawled_child.lang != langid:
        logging.info(f"Crawled child was in language {crawled_child.lang} but wanted to crawl {langid}")
    logging.info(f'Crawling child with url {crawled_child.url}')

    new_state = State(languages=state.languages, link_queue_limit = state.link_queue_limit)

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
        if environment.isParallel(env, crawled_doc, monolingual_doc):
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


def create_start_state_from_node(root, languages, link_queue_limit):
    state = State(languages, link_queue_limit=link_queue_limit)

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
    


############################################################
#                     STATE MEMBERS                        # 
############################################################

class Link():
    def __init__(self, link_text: str,
                        link_text_lang: int,
                        link_url: str,
                        link_url_id: int):
        self.link_text = link_text
        self.language = link_text_lang
        self.url = link_url
        self.id = link_url_id

class MonolingualDocument():
    def __init__(self, docid, langid):
        self.docid = docid
        self.language = langid


class ParallelDocument:
    def __init__(self, id=None):
        self.parallel_id = id
        self.docids = set()
    
    def add_document(self, id):
        self.docids.add(id)

class RewardHistory:
    def __init__(self):
        pass

class DecisionHistory:
    def __init__(self):
        pass