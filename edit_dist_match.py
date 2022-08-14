import editdistance
import sys

# Given a candidate url, return the index of the item in existing_urls with the smallest string edit distance.
def FindMinEditDistance(candidate_url, existing_urls):
    distances = [editdistance.eval(candidate_url, curr) for curr in existing_urls]
    min_dist = sys.maxsize 
    min_dist_idx = -1
    for i in range(len(distances)):
        if distances[i] < min_dist:
            min_dist = distances[i]
            min_dist_idx = i
    return min_dist #min_dist_idx



a = FindMinEditDistance("https://www.visitbritain.com/gb/en/plan-your-trip/getting-around-britain/travelling-coach", ["https://www.visitbritain.com/fr/fr/preparez-votre-voyage/transports-en-grande-bretagne/se-deplacer-en-bus"])
print(a)

a = FindMinEditDistance("https://www.buchmann.ch/en/schaffner-sntis-table-140x80-cm-fir-green-p-409164.html",
["https://www.buchmann.ch/en/schaffner-sntis-table-140x80-cm-fir-green-p-409164.html?language=fr"])
print(a)
