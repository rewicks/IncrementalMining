
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

# Compute inverse counts (total - my count) for each input count. This is our numerator.
# Sum those to get a new denominator.
class InverseCountDecider(Decider):
    def Decide(self, counts):
        # Assumption: final count is "other", and we don't care about it.
        my_counts=counts[:-1]
        total=sum(my_counts)
        diffs=[total - c for c in my_counts]
        diffsum = sum(diffs)
        ret = [d / diffsum for d in diffs]
        # Make sure we're returning something that looks like a probability distribution
        assert(abs(1.0 - sum(ret)) < .01)
        return ret


