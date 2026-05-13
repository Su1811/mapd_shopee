from common_online import OnlineGraphPolicySolver


class VRPOrToolsSolver(OnlineGraphPolicySolver):
    """
    Online dynamic VRP-style solver.

    Không giả định biết toàn bộ orders từ đầu. Ở mỗi step, nó thực hiện insertion
    heuristic trên các đơn đã reveal. Tên file/class giữ nguyên để tương thích grader.
    """

    def __init__(self, env):
        super().__init__(env, policy_name="vrp")
