from state import *
from decider import *


def trajectory(env, langIds, linkQueueLimit, algorithm, maxStep, quiet, coeffs, gamma):
    state = create_start_state_from_node(env.rootNode, langIds, linkQueueLimit)

    ep_reward = 0
    discount = 1.0
    docs = []
    if algorithm == 'random':
        decider = RandomDecider()
    elif algorithm == 'linear':
        decider = LinearDecider(coeffs)
    else:
        decider = None

    for t in range(1, maxStep):
        probs = decider.CalcProbs(state)
        if probs is not None:
            link = decider.ChooseLink(state, probs)
            assert(link is not None)

            new_state = transition_on_link(env, state, link)
            assert(new_state is not None)
        else:
            break

        # logging of stats for comparing models (not used for training)
        reward = get_reward(state, new_state)
        docs.append(len(new_state.parallel_documents))
        ep_reward += discount * reward
        discount *= gamma
        ##############################################################

        state = new_state

    #if not quiet:
    #    print(f"Documents: {','.join([str(x) for x in docs])}")

    crawl_histories = []
    for x in range(1):
        crawl_histories.append(docs)
    
    if not quiet:
        print(f"Reward: {ep_reward}")
        print("CRAWL HISTORIES")
        for x in crawl_histories:
            print(",".join([str(y) for y in x]))
        
        print("AUC of CRAWL HISTORIES")
        for x in crawl_histories:
            print(sum(x))

        print("Mean AUC for all crawl histories: ", np.mean([np.sum(x) for x in crawl_histories]))
    auc = sum(docs)
    return ep_reward, auc, docs

def trajectories(envs, langIds, linkQueueLimit, algorithm, maxStep, quiet, coeffs, gamma):
    for host, env in envs:
        trajectory(env, langIds, linkQueueLimit, algorithm, maxStep, quiet, coeffs, gamma)
