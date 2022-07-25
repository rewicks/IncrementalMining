import numpy as np
import scipy.special


class Decider:
    def ChooseLink(self, state, probs):
        assert(len(state.link_queue) > 0)
        link = np.random.choice(state.link_queue, 1, p=probs)
        #print("link", link)
        return link[0]

class RandomDecider(Decider):
    def CalcProbs(self, state):
        if len(state.link_queue) > 0:
            ret = np.empty([len(state.link_queue)])
            ret.fill(1./len(state.link_queue))
            return ret
        else:
            return None

class LinearDecider(Decider):
    def __init__(self, coeffs):
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
        # langCosts = scipy.special.softmax(langCosts)

        probs = np.empty([len(state.link_queue)])
        for linkIdx, link in enumerate(state.link_queue):
            cost = np.zeros([6])
            #print(link.language) 

            if link.language == state.languages[0]:
                cost[0] = features[0] - features[1]
                cost[1] = features[3] - features[4]
            elif link.language == state.languages[1]:
                cost[0] = features[1] - features[0]
                cost[1] = features[4] - features[3]

            if link.parent_lang == state.languages[0]:
                cost[2] = features[0] - features[1]
                cost[3] = features[3] - features[4]
            elif link.parent_lang == state.languages[1]:
                cost[2] = features[1] - features[0]
                cost[3] = features[4] - features[3]
            
            if link.link_text in ["en", "Eng", "English", "Anglais"]:
                cost[4] = features[1] - features[0]
                cost[5] = features[3] - features[4]
            elif link.link_text in ["fr", "Fra", "Français", "Francais", "Französisch"]:
                cost[4] = features[0] - features[1]
                cost[5] = features[4] - features[3]


            # # anchor text
            # if link.link_text in ["en", "Eng", "English", "Anglais", "fr", "Fra", "Français", "Francais"]:
            #     costs[6] = len(state.link_queue)

            #print("costs", costs)
            linkCost = np.inner(self.coefficients, cost)
            #print("linkCost", linkCost)
            probs[linkIdx] = linkCost

        if len(probs) == 0:
            return None
        probs = scipy.special.softmax(probs)
        #print("probs", probs.shape, np.sum(probs))
        return probs
