import torch.nn as nn
import copy


class EarlyExitNetworkSegmentor(nn.Module):
        def __init__(self,network,exit=None,threshold=None,device='cpu',scaler=None):
            super(EarlyExitNetworkSegmentor, self).__init__()
            self.network = copy.deepcopy(network)
            self.exit = copy.deepcopy(exit)
            self.threshold = threshold
            self.device=  device
            self.scaler = scaler
                
        def forward(self,x):
            x = x.to(self.device)
            x = x.unsqueeze(0)
            x = self.network(x)
            if self.exit != None:
                early_exit = self.exit(x)
                early_exit=  early_exit.squeeze(0)
                early_exit = self.scaler(early_exit)
                if early_exit.max() > self.threshold:
                            return early_exit,True
                return x.squeeze(0),False
            x = x.squeeze(0)
            return x,True


def main():
    return  

if __name__ == "main":
    main()