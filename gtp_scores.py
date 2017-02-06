# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict

import numpy as np

from utils import sample_from_list


class GTPScores(object):
    def __init__(self, ground_truth_pairs):
        self.gtp_max_precisions = OrderedDict([
            (gtp, 0) for gtp in ground_truth_pairs
        ])

    @property
    def ground_truth_pairs(self):
        return list(self.gtp_max_precisions)

    @property
    def remaining_gain(self):
        return len(self.gtp_max_precisions) - self.score

    @property
    def score(self):
        return sum([mp for mp in self.gtp_max_precisions.values()])

    def get_remaining_gain_for(self, gtp):
        return 1 - self.gtp_max_precisions[gtp]

    def get_remaining_gains(self):
        return OrderedDict([
            (gtp, 1 - mp)
            for gtp, mp in self.gtp_max_precisions.items()
        ])

    def copy_reset(self):
        return GTPScores(self.ground_truth_pairs)

    def update_with_gps(self, gps):
        """Update with list of graph patterns and return precision gain."""
        precision_gain = 0
        for gp in gps:
            for gtp, precision in gp.gtp_precisions.items():
                old = self.gtp_max_precisions[gtp]
                if precision > old:
                    precision_gain += precision - old
                    self.gtp_max_precisions[gtp] = precision
        return precision_gain

    def remaining_gain_sample_gtps(self, max_n=None):
        """Sample ground truth pairs according to remaining gains.

        This method draws up to max_n ground truth pairs using their remaining
        gains as sample probabilities. GTPs with remaining gain of 0 are never
        returned, so if less than n probabilities are > 0 it draws less gtps.

        :param max_n: Up to n items to sample.
        :return: list of ground truth pairs sampled according to their remaining
            gains in gtp_scores with max length of n.
        """
        gtps, gains = zip(*self.get_remaining_gains().items())
        return sample_from_list(gtps, gains, max_n)

    def __sub__(self, other):
        if not isinstance(other, GTPScores):
            raise TypeError('other should be GTPScore obj as well')
        if self.ground_truth_pairs != other.ground_truth_pairs:
            raise TypeError("can't compare GTPScores over different gtps")
        return np.sum(
            np.array(self.gtp_max_precisions.values()) -
            np.array(other.gtp_max_precisions.values())
        )
