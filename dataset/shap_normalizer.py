import torch


class ShapeNormalizer:
    """
    Single source of truth for tensor shapes throughout the pipeline.

    Contracts:
        raw      : [N]       — flat time series from DatasetBuilder
        flat     : [N, 1]    — after unsqueeze, before sequencing
        sequence : [B, L, C] — after SequenceBuilder, always 3D
        pred     : [B, P, 1] — forecaster output, always 3D, single channel
    """

    # ------------------------------------------------------------------
    # Y: raw → flat
    # ------------------------------------------------------------------
    @staticmethod
    def y_to_flat(Y: torch.Tensor) -> torch.Tensor:
        """[N] or [N, 1] → [N, 1]"""
        if Y.dim() == 1:
            return Y.unsqueeze(-1)
        if Y.dim() == 2 and Y.shape[1] == 1:
            return Y
        raise ValueError(f"Unexpected Y shape for y_to_flat: {Y.shape}")

    # ------------------------------------------------------------------
    # Y: sequence → [B, P] for loss computation
    # ------------------------------------------------------------------
    @staticmethod
    def y_seq_to_2d(Y: torch.Tensor) -> torch.Tensor:
        """[B, P, 1] or [B, P] → [B, P]"""
        if Y.dim() == 3:
            return Y[..., 0]
        if Y.dim() == 2:
            return Y
        raise ValueError(f"Unexpected Y shape for y_seq_to_2d: {Y.shape}")

    # ------------------------------------------------------------------
    # Forecaster output → [B, P] for loss computation
    # ------------------------------------------------------------------
    @staticmethod
    def pred_to_2d(preds: torch.Tensor) -> torch.Tensor:
        """[B, P, C] or [B, P] → [B, P] (always takes first channel)"""
        if preds.dim() == 3:
            return preds[..., 0]
        if preds.dim() == 2:
            return preds
        raise ValueError(f"Unexpected preds shape for pred_to_2d: {preds.shape}")

    # ------------------------------------------------------------------
    # Forecaster output → [B, P, 1] for storage / tracking
    # ------------------------------------------------------------------
    @staticmethod
    def pred_to_3d(preds: torch.Tensor) -> torch.Tensor:
        """[B, P, C] or [B, P] → [B, P, 1]"""
        if preds.dim() == 3:
            return preds[..., 0:1]
        if preds.dim() == 2:
            return preds.unsqueeze(-1)
        raise ValueError(f"Unexpected preds shape for pred_to_3d: {preds.shape}")

    # ------------------------------------------------------------------
    # Sanity check — call at pipeline boundaries
    # ------------------------------------------------------------------
    @staticmethod
    def assert_seq_shape(name: str, t: torch.Tensor, expected_C: int = None):
        assert t.dim() == 3, f"{name} must be 3D [B, L, C], got {t.shape}"
        if expected_C is not None:
            assert t.shape[2] == expected_C, (
                f"{name} expected C={expected_C}, got {t.shape}"
            )