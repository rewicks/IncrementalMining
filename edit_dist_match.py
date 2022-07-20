import editdistance
import sys

# Given a candidate url, return the index of the item in existing_urls with the smallest string edit distance.
def FindStrandMatch(candidate_url, existing_urls):
    distances = [editdistance.eval(candidate_url, curr) for curr in existing_urls]
    min_dist = sys.maxsize 
    min_dist_idx = -1
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
            min_dist_idx = i
    return min_dist_idx



