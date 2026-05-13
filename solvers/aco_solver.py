from common_online import OnlineGraphPolicySolver


class ACOSolver(OnlineGraphPolicySolver):
    """
    Online ACO-style solver.

    Pheromone được cập nhật từ lịch sử đơn đã xuất hiện, nên không leak hotspot/order tương lai.
    """

    def __init__(self, env):
        super().__init__(env, policy_name="aco")
