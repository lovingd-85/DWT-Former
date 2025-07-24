import torch
import torch.nn as nn     

class bone_block(nn.Module):
    def __init__(self,d_model=16, input_len=96 ,layers=5):
        super().__init__()
        self.block = nn.ModuleList(
            [
                c2f_block(cov_num=3,d_model=d_model) for i in range(layers)
            ]
        )
        cat_num = 2*input_len-1
        for i in range(layers-1):
            cat_num += input_len//(2**i)

        self.L1 = nn.Linear(cat_num, input_len)
        self.act = nn.GELU()
        self.L2 = nn.Linear(input_len, input_len)
        self.norm = nn.LayerNorm(input_len)
    def forward(self, x):
        # for i in range(len(self.block)):
        #     x[i] = self.block[i](x[i])

        x = torch.cat(x,dim=-1)
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        x = self.norm(x).transpose(1,2)
        return x

        

class c2f_block(nn.Module):
    def __init__(self, cov_num, d_model=16):
        super().__init__()     
        self.cov_list = nn.ModuleList()
        for i in range(cov_num):
           self.cov_list.append(
                nn.Conv1d(d_model//2, d_model//2, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
           )
        self.norm_half = nn.LayerNorm(d_model//2)
        self.linear = nn.Linear(d_model//2+d_model, d_model)  
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
          
    def forward(self, x):
        _ , _ , D = x.shape
        ox , px= torch.split(x, D//2, dim=-1)
        covx = px.transpose(1,2)
        for layer in self.cov_list:
            covx = layer(covx)
            covx = self.act(self.norm_half(covx.transpose(1,2))).transpose(1,2)
        x = torch.cat([ox,px,covx.transpose(1,2)],dim=-1)
        x = self.act(self.norm(self.linear(x)))
        return x


class seq2pre(nn.Module):
    def __init__(self,seq_len,pred_len,d_model,c_out):
        super().__init__()
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear1 = nn.Linear(seq_len,pred_len)
        self.Linear2 = nn.Linear(pred_len,pred_len)
        # self.Linear3 = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(pred_len)
        self.normc = nn.LayerNorm(pred_len)
        self.act = nn.GELU()


    def forward(self,x):
        x = self.act(self.norm(self.Linear1(x)))
        x = self.act(self.norm(self.Linear2(x))).transpose(1,2)
        # x = self.act(self.normc(self.Linear3(x).transpose(1,2)))
        return x