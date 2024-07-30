import torch
from torch import nn

class MovingAverageEarlyStopper(nn.Module):
    def __init__(self, patience, ma_alpha = 0.9):
        super(MovingAverageEarlyStopper, self).__init__()
        self.patience = nn.Parameter(torch.tensor(patience), requires_grad=False)
        self.best_loss = nn.Parameter(torch.tensor(float('inf')), requires_grad=False)
        self.best_step = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.loss_ma = nn.Parameter(torch.tensor(float('nan')), requires_grad=False)
        self.ma_alpha = nn.Parameter(torch.tensor(ma_alpha), requires_grad=False)

    def forward(self, loss, step):
        loss = torch.tensor(loss)

        if torch.isnan(self.loss_ma):
            self.loss_ma.data = loss
        else:
            self.loss_ma.data = self.loss_ma * self.ma_alpha + loss * (1 - self.ma_alpha)

        new_best = self.loss_ma < self.best_loss
        if new_best:
            self.best_loss.data = self.loss_ma
            self.best_step.data = torch.tensor(step).float()

        return torch.tensor(self.loss_ma), new_best, step - self.best_step > self.patience
