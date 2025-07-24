import torch
import torch.nn as nn
from models.bone import bone_block, seq2pre, c2f_block
from models.mutihaedattation import Attation_Cov
from models.multiscalemix import Multibolck, DeMultibolck
import numpy as np
import matplotlib.pyplot as plt
from models.attation import Encoder, EncoderLayer, AttentionLayer, FullAttention
from models.embed import DataEmbedding_inverted

class DWTmodel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.multi_bolck_one = Multibolck(args.seq_len ,args.d_model, args.freq, args.dropout, args.sampling_layers,args.d_ff,time_mix = True)
        self.multi_bolck_two = Multibolck(args.seq_len ,args.d_model, args.freq, args.dropout, args.sampling_layers,args.d_ff,time_mix = False)


        self.featur_bedding = nn.Sequential(
            nn.Linear(args.c_in-1, 512),
            nn.GELU(),
            nn.Linear(512,512)
        )

        self.channel = c2f_block(3, d_model=512)

        self.enc_embedding = DataEmbedding_inverted(args.seq_len, 128, 'timeF', 24,
                                                    0.1)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=5, attention_dropout=0.1,
                                      output_attention=
                                      True), 128, 8),
                    d_model = 128,
                    d_ff = 128,
                    dropout=0.1,
                    activation='gelu'
                ) for l in range(2)
            ],
            norm_layer=torch.nn.LayerNorm(128)
        )
        self.treatline = nn.Linear(6,32)


        self.demot = DeMultibolck(args.d_model, args.seq_len, args.sampling_layers)

        self.seq2pre_x = seq2pre(args.seq_len, args.pred_len, args.d_model, args.c_out)

        treat = {'h':4,'t':5}
        self.line= nn.Linear(args.c_in + treat[args.freq] + args.d_model, args.d_ff)
        self.Learn2 = nn.Linear(args.d_ff, args.d_ff)
        self.normd_mode = nn.LayerNorm(args.d_ff)

        self.normd_ff = nn.LayerNorm(args.d_ff)
        self.act = nn.GELU()
        self.projection = nn.Linear(128, args.pred_len)
        self.prodrop = nn.Dropout(0.1)
        self.projection2 = nn.Linear(args.d_ff, args.c_out)



    def forward(self, orx, orx_mark):
        means = orx.mean(1, keepdim=True).detach()
        orx = orx - means
        stdev = torch.sqrt(torch.var(orx, dim=1, keepdim=True, unbiased=False) + 1e-5)
        orx /= stdev

        # Wave 
        _, _, N = orx.shape
        orx_target = orx[:,:,-1]
        orx_feature = orx[:,:,:-1]
        out_one_list, out_one = self.multi_bolck_one(orx_target, orx_mark)
        out_two_list, out_two = self.multi_bolck_two(out_one.squeeze(-1).detach(), orx_mark)
        
        out_list = []
        for o, t in zip(out_one_list, out_two_list):
            out_list.append(o + t)
        
        out = self.demot(out_list)

        out = self.seq2pre_x(out.transpose(1,2))

        enc_out = self.enc_embedding(orx, orx_mark)
        enc_out, attns = self.encoder(enc_out)

        dec_out = self.projection(enc_out).permute(0, 2, 1)
        

        dec_out = torch.cat([out,dec_out],dim=-1)
        dec_out = self.act(self.normd_mode(self.line(dec_out)))

        dec_out = self.projection2(dec_out)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.args.pred_len, 1))
        result = dec_out

        return result

        
    def __up_sample(self, x):
        device = x.device
        B,L,D = x.shape
        up_x = torch.empty(B, L*2-1, D).to(device)
        mid =(x[:,:-1,:] + x[:,1:,:])/2
        up_x[:,::2,:] = x
        up_x[:,1::2,:] = mid
        x = up_x
        return x
    
    def __draw_atten(self, atta):
        save_att = "/home/qihui/EXP/Mymodel_v2/view/atta"
        score = atta[1][5,:,:,:].to('cpu').detach().numpy()
        H, w, w = score.shape
        pics, axs = plt.subplots(2,4,figsize=(12, 6))
        for i in range(H):
            matrix = score[i]
            ax = axs[i//4][i%4]
            ax.imshow(matrix, cmap='viridis', interpolation='nearest')  # cmap 是颜色映射，可以选择不同的颜色映射
            if i == 0:
                pics.colorbar(ax.images[0], ax=ax)  # 添加颜色条，用于表示矩阵值与颜色的对应关系
            plt.title("attation score")  # 添加标题
        plt.savefig(save_att+f"/score"+".jpg")


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        m = torch.cat([front, x, end], dim=1)
        m = self.avg(m.permute(0, 2, 1))
        m = m.permute(0, 2, 1)
        res = x - m
        return res, m 

