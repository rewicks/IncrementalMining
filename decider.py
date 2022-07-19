
class Decider:
  def Decide(self, counts):
    sum = 0
    for count in counts:
        sum += count

    ret = []
    for i in range(len(counts)):
        val = counts[i] / sum
        ret.append(val)
    return ret
