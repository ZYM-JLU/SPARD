import torch.nn as nn
from models.refine_model import np_GCN


class NoisePredictor(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48, k = 50,np_alpha = 3.0):
        super().__init__()
        self.k = k
        output_feature = input_feature * k
        self.model = np_GCN(input_feature,hidden_feature,output_feature,p_dropout,num_stage,node_n)
        self.np_alpha = np_alpha

    def forward(self,past_dct):
        past_dct = past_dct.permute(0,2,1).contiguous()
        noise = self.model(past_dct,noise = self.np_alpha)
        noise = noise.reshape(noise.size(0),noise.size(1),self.k,-1).permute(0,2,3,1).contiguous()
        return noise
