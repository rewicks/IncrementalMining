# IncrementalMining
Systems such as Paracrawl/Bitextor batch process each step of the pipeline, ie. a crawler downloads many pages for each website, a document aligner searches for parallel documents in the set of pages, and so forth.

We propose using a reinforcement learning approach to decide which pages to crawl and when to stop crawling in order to optimize time and resources.


## Environment
----

Incremental Mining uses a simulated environment of an actual web crawler. A handful of domains have been pre-crawled with data extracted to interact with decision making process of Incremental Mining.

To interact with this environment, we create the `SimulatedEnvironment()` class.


## Crawler State
----

Incrementally expands a domain's tree with a reinforcement learning based decision-making process.

The state of the crawler is defined by:

1. Queue of `Links`: The queue is limited by size `k.` At each decision-making step. The crawler can decide to download one of these links. When a new link is pushed (after appearing on a crawled page), old links are popped to maintain queue size.

2. List of unprocessed `MonolingualDocument`s: the text from all crawled pages. It has not yet been aligned.

3. List of `ParallelDocument`s: list of all documents (and sentences) that have been extracted from crawler. This can be bitext or n-ways parallel. In practice, this will probably just be metadata about them (length, languages, etc).

4. `RewardHistory`: a list of length `h` on the return/reward of previous decisions. This will inform the stopping mechanism.

5. `DecisionHistory`: a list of previous decisions

## Smart Crawler Actions
---
At each step, the crawler can make one of three decisions:

1. Crawl a child link from the queue (does not have to be the first item in queue)

2. Increment (the loop element of diagram) by processing the `unprocessed documents` via document alignment, sentence alignment, and filtering.

3. Stop processing


