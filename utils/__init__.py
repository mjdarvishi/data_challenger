import typing
if typing.TYPE_CHECKING:
    from core.config import XFeature


def cal_input_dimenion(features: list["XFeature"]) -> int:
    num_features = len(features)
    # Layout: [hour_idx, x1..xN]
    return 1 + num_features