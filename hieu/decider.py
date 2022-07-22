import numpy as np
import scipy.special


class Decider:
    def ChooseLink(self, state, probs):
        if len(state.link_queue) > 0:
            link = np.random.choice(state.link_queue, 1, p=probs)
            #print("link", link)
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
    def __init__(self, coeffs = np.array([5, 5, 5, 5, 5, 5, 5])):
        self.coefficients = coeffs

    def ChooseLink(self, state, probs):
        if len(state.link_queue) > 0:
            link = state.link_queue[np.argmax(probs)]
            #print("link", link)
            return link
        else:
            return None

    def CalcProbs(self, state):
        #print("self.languages", state.languages)
        features = state.get_features()
        #features = np.array(self.get_features())
        #print("features", features)

        langCosts = features[:3]
        langCosts = scipy.special.softmax(langCosts)

        probs = np.empty([len(state.link_queue)])
        for linkIdx, link in enumerate(state.link_queue):
            costs = np.zeros([7])
            #print(link.language) 
            if link.language == state.languages[0]:
                costs[0] = langCosts[0]
            elif link.language == state.languages[1]:
                costs[1] = langCosts[1]
            else:
                costs[2] = langCosts[2]

            if link.parent_lang == state.languages[0]:
                costs[3] = langCosts[0]
            elif link.parent_lang == state.languages[1]:
                costs[4] = langCosts[1]
            else:
                costs[5] = langCosts[2]

            # anchor text
            if link.link_text in ["en", "Eng", "English", "Anglais", "fr", "Fra", "Français", "Francais"]:
                costs[6] = len(state.link_queue)

            #print("costs", costs)
            linkCost = np.inner(self.coefficients, costs)
            #print("linkCost", linkCost)
            probs[linkIdx] = linkCost

        if len(probs) == 0:
            return None
        probs = scipy.special.softmax(probs)
        #print("probs", probs.shape, np.sum(probs))
        return probs