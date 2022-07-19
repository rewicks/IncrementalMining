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

# Base class for all other states
class State():
    def __init__(self, link_queue_limit=1000):

        # unless people protest I think a list of size k is
        # actually simpler than a queue
        self.link_queue = []
        self.link_queue_limit = link_queue_limit

        # these actually might be *too* large to keep, so maybe
        # just keep metadata from them
        self.monolingual_documents = []
        self.parallel_documents = []

    def add_link(self, link):
        self.link_queue.append(link)
        if len(self.link_queue) > self.link_queue_limit:
            self.link_queue.pop(0)

    
def transition_on_lang_prob(env, state, decision, langIds):
    # langid = torch.argmax(decision, dim=0)
    langid = langIds[torch.argmax(decision, dim=0)]

    link_to_crawl = None
    for li in state.link_queue:
        if li.linkLang == langid:
            link_to_crawl = li
            break
    
    if link_to_crawl is None:
        logging.info(f"Crawler decided to crawl {langid} but no documents in this language remained in link queue.")
        return None
    
    crawled_child = env.crawl_child(link_to_crawl.id)

    new_state = State(link_queue_limit = state.link_queue_limit)
    for li in state.link_queue:
        if li.url != link_to_crawl.url:
            new_state.add_link(li)
    
    for child_link in crawled_child.links:
        link = Link(link_text=child_link.text,
                    link_text_lang=child_link.textLang,
                    link_url=child_link.childNode.url,
                    link_url_id=child_link.childNode.urlId)
        new_state.add_link(link)

    for parallel_doc in state.parallel_documents:
        new_state.append(parallel_doc)

    crawled_doc = MonolingualDocument(docid=crawled_child.urlId)
    aligned = False
    for monolingual_doc in state.monolingual_documents:
        if environment.isParallel(env, crawled_doc, monolingual_doc):
            parallel_doc = ParallelDocument(len(new_state.parallel_documents))
            parallel_doc.add_document(crawled_doc.docid)
            parallel_doc.add_document(monolingual_doc.docid)
            new_state.append(parallel_doc)
            aligned = True
        else:
            new_state.monolingual_documents.append(monolingual_doc)

    if not aligned:
        new_state.monolingual_documents.append(crawled_doc)
    return new_state


def create_start_state_from_node(root, link_queue_limit=1000):
    state = State(link_queue_limit=link_queue_limit)

    # update with the only crawled page
    root_document = MonolingualDocument(docid=root.urlId)
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
        self.linkLang = link_text_lang
        self.url = link_url
        self.id = link_url_id

class MonolingualDocument():
    def __init__(self, docid):
        self.docid = docid

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