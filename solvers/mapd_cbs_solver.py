from common_online import OnlineGraphPolicySolver


class MAPDCBSSolver(OnlineGraphPolicySolver):
    """
    Online MAPD-CBS-style solver.

    Dùng BFS trên đồ thị lưới và phạt xung đột mục tiêu gần nhau khi phân công.
    Collision thật vẫn được môi trường xử lý theo ưu tiên shipper id nhỏ hơn.
    """

    def __init__(self, env):
        super().__init__(env, policy_name="cbs")
