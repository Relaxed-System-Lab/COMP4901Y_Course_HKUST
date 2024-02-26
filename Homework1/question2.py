import torch
import torch.nn as nn
from typing import List


# ==========================
seed = 2024
torch.manual_seed(seed)

iters = int(1e5)
N, D, H = 500, 64, 64
lr = 1e-1
# ==========================


class MLP:
    def __init__(self, in_features, out_features, lr):
        self.lr = lr

        self.W1 = torch.randn(in_features, out_features)
        self.W2 = torch.randn(out_features, 1)

        self.dW1 = None
        self.dW2 = None

    
    def forward(self, x, y_label)-> torch.Tensor:
        # =========================================
        # Your code here. 
        pass
        # =========================================

    
    def backward(self, loss):
        # =========================================
        # Your code here. 
        pass
        # =========================================


    def sgd_step(self):
        # =========================================
        # Your code here. 
        pass

        # =========================================


def plot(losses: List):

    # =========================================

    # insert your code here to visualize the result matplotlib is needed. 
    pass

    # =========================================


if __name__ == "__main__":
    losses = []

    x = torch.randn(N, D)
    y = torch.randn(N, 1)
    mlp = MLP(D, H, lr)


    for iter in range(iters):
        sample_id = torch.randint(0, N, size=(1, )).item()

        input = x[sample_id, :].unsqueeze(0)
        y_label = y[sample_id]

        loss = mlp.forward(input, y_label)

        mlp.backward(loss)

        mlp.sgd_step()


        if iter % 100 == 0: 
            losses.append((iter, loss))

    with open('./Q2_output.txt', 'w') as f:
        for iter, loss in losses:
            f.write(f"At {iter}-th iteration, loss is {loss}\n")
                

    plot(losses)




