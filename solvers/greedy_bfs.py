from common_online import OnlineGraphPolicySolver


class GreedyBFS(OnlineGraphPolicySolver):
    """Greedy BFS online baseline: assign visible orders by reward/deadline/travel score."""

    def __init__(self, env):
        super().__init__(env, policy_name="greedy")
