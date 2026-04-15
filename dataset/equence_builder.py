import torch


class SequenceBuilder:
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len

    def build(self, X, Y):
        X_seq = []
        Y_seq = []

        total = X.shape[0]

        for i in range(total - self.seq_len - self.pred_len):
            x_window = X[i : i + self.seq_len]

            y_window = Y[i + self.seq_len : i + self.seq_len + self.pred_len]

            if y_window.dim() == 1:
                y_window = y_window.unsqueeze(-1)

            X_seq.append(x_window)
            Y_seq.append(y_window)

        return torch.stack(X_seq), torch.stack(Y_seq)
