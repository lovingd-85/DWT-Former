import torch
import torch.nn as nn
import numpy as np
import pywt 
from models.embed import DataEmbedding

class DeMultibolck(nn.Module):
    def __init__(self, d_model,seq_len, sampling_layers):
        super().__init__()
        self.chan2pre = nn.ModuleList(
            [nn.Linear(seq_len//(2**i), seq_len) for i in range(sampling_layers + 1)]
        )
        self.act = nn.GELU()
        self.pre2pre = nn.ModuleList(
            [nn.Linear(seq_len, seq_len) for i in range(sampling_layers + 1)]
        )
        self.project = nn.Linear(d_model*(sampling_layers+1), d_model)

    def forward(self, x):
        out_list = []
        for i in range(len(x)):
            o = self.pre2pre[i](self.act(self.chan2pre[i](x[i].transpose(1,2)))).transpose(1,2)
            out_list.append(o)
        out = self.project(torch.cat(out_list, dim=-1))
        
        return out


class Multibolck(nn.Module):
    def __init__(self, seq_len ,d_model, freq, dropout, sampling_layers, d_ff,time_mix = True):
        super().__init__()
        self.sampling_layers = sampling_layers
        self.time_mix = time_mix
        self.embedding_t = DataEmbedding(1, d_model , 'fixed', freq, dropout)
        self.embedding_s = DataEmbedding(1, d_model , 'fixed', freq, dropout)
        self.line_bedding_t = nn.Linear(1,d_model)
        self.line_bedding_s = nn.Linear(1,d_model )
        self.act = nn.GELU()
        self.mutscatredm = MultiScaleTrendMixing(seq_len, sampling_layers)
        self.mutscaseam = MultiScaleSeasonMixing(seq_len, sampling_layers)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )
        self.project = nn.Linear(1, d_model)
        self.nome = nn.LayerNorm(d_model)
        self.d_mode12one = nn.Linear(d_model, 1)
    def forward(self, orx_target, orx_mark):
        # pic, ax = plt.subplots()
        # 对orx_targer进行小波分解，分离为trend,season
        trend_list = []
        season_list = []
        B,T = orx_target.shape
        # '''消融实验
        device = orx_target.device
        orx_target = orx_target.to('cpu')
        
        # ax.plot([i for i in range(len(orx_treate[1]))],orx_target[1])
        
        for i in range(B):
            t, s  = self.wavelet_decompose(orx_target[i,:], wavelet='dmey', level=5) #原本是‘db4’   
            trend_list.append(t)
            season_list.append(s)
        # ax.plot([i for i in range(len(trend_list[1]))], trend_list[1] )
        # ax.plot([i for i in range(len(orx_treate[1]))], season_list[1])
        # pic.savefig("ts.png")
        trend = torch.asarray(trend_list).unsqueeze(-1).to(device)
        season = torch.asarray(season_list).unsqueeze(-1).to(device)


        if self.time_mix:
            trend = self.embedding_t(trend, orx_mark)  # [B,T,C]
            season = self.embedding_s(season, orx_mark)
        else:
            trend = self.act(self.line_bedding_t(trend))
            season = self.act(self.line_bedding_s(season))
        

        #多尺度处理
        trend, season, x_mark = self.__multi_scale_process_inputs('avg',trend, season, orx_mark )

        trend = self.mutscatredm(trend)
        season = self.mutscaseam(season)        
        # '''

        # if self.time_mix:
        #     out = self.embedding_s(orx_target.unsqueeze(-1), orx_mark)
        # else:
        #     out = self.act(self.line_bedding_t(orx_target.unsqueeze(-1)))

        # out1, out2, x_mark = self.__multi_scale_process_inputs('avg',out, out, orx_mark )

        out_list = []
        ori = orx_target.unsqueeze(-1).to(device)
        # ori = orx_target.unsqueeze(-1)
        for out_season, out_trend in zip(trend, season):
            out = out_season + out_trend
            out_list.append(out)
        # for o1, o2 in zip(out1, out2):
        #     out = o1 + o2
        #     out_list.append(out)
        ori2d_model = self.nome(self.project(ori))
        out_list[0] = out_list[0] + ori2d_model
        result = self.d_mode12one(out_list[0])
        return out_list, result


    def __multi_scale_process_inputs(self, down_sampling_method,trend, season, x_mark):
        trend_sampling_list = []
        season_sampling_list = []
        x_mark_sampling_list = []


        if down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(2)
        elif down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(2)
        elif down_sampling_method == 'conv':
            down_pool = nn.Conv1d(in_channels=7, out_channels=16,
                                  kernel_size=3,stride=2,padding_mode='circular',bias=False)
        else:
            assert False, 'down_sampling_method must be max, avg or conv'

        trend_sampling_list.append(trend)
        season_sampling_list.append(season)
        x_mark_sampling_list.append(x_mark)
        for i in range(self.sampling_layers):
            trend_sampling = down_pool(trend.permute(0,2,1)).permute(0,2,1)
            season_sampling = down_pool(season.permute(0,2,1)).permute(0,2,1)

            trend_sampling_list.append(trend_sampling)
            season_sampling_list.append(season_sampling)
            trend = trend_sampling
            season = season_sampling

            if x_mark is not None:
                x_mark_sampling_list.append(x_mark[:, ::2, :])
                x_mark = x_mark_sampling_list[-1]

        trend = trend_sampling_list
        season = season_sampling_list
        if x_mark is not None:
            x_mark= x_mark_sampling_list
        else:
            x_mark = x_mark

        return trend, season, x_mark


    #去噪
    def wavelet_denoise(self, signal, wavelet='db4', level=3, threshold_method='soft'):
        """
        使用小波变换对信号进行去噪。
        
        参数:
        - signal: 输入信号（一维数组）
        - wavelet: 小波基函数，默认 'db4'（Daubechies 4）
        - level: 分解层数，默认 3
        - threshold_method: 阈值方法，可选 'soft' 或 'hard'
        
        返回:
        - denoised_signal: 去噪后的信号
        """
        # 小波分解
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # 计算阈值（通用阈值公式）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 估计噪声标准差
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # 对细节系数进行阈值处理
        denoised_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # 保留近似系数
                denoised_coeffs.append(coeff)
            else:  # 对细节系数进行阈值处理
                if threshold_method == 'soft':
                    denoised_coeffs.append(pywt.threshold(coeff, value=threshold, mode='soft'))
                elif threshold_method == 'hard':
                    denoised_coeffs.append(pywt.threshold(coeff, value=threshold, mode='hard'))
                else:
                    raise ValueError("Invalid threshold_method. Choose 'soft' or 'hard'.")
        
        # 小波重构
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        return denoised_signal

    # 小波分解函数
    def wavelet_decompose(self, data, wavelet='db4', level=5):
        """
        使用小波变换分解时间序列，分离出趋势和季节性。
        
        参数:
        - data: 输入时间序列（一维数组）
        - wavelet: 小波基函数，默认 'db4'（Daubechies 4）
        - level: 分解层数，默认 5
        
        返回:
        - trend: 趋势部分（低频成分）
        - seasonal: 季节性部分（高频成分）
        """
        # 小波分解
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # 近似系数表示趋势（低频成分）
        trend = pywt.waverec([coeffs[0]] + [None] * level, wavelet)
        
        # 细节系数表示季节性和噪声（高频成分）
        seasonal_coeffs = [None] + coeffs[1:] + [None] * (len(coeffs) - level - 1)
        seasonal = pywt.waverec(seasonal_coeffs, wavelet)
        
        # 截断长度以匹配原始数据长度（由于边界效应）
        trend = trend[:len(data)]
        seasonal = seasonal[:len(data)]
        
        return trend, seasonal
    
class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """
    def __init__(self, seq_len, sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (2 ** i),
                        seq_len // (2 ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (2 ** (i + 1)),
                        seq_len // (2 ** (i + 1)),
                    ),

                )
                for i in range(sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        if len(season_list) == 1:
            out_low = out_high = season_list[0]
        else:
            out_high = season_list[0]
            out_low = season_list[1]
        out_season_list = [out_high]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high.permute(0, 2, 1)).permute(0, 2, 1)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high)
        # self.Draw_data()
        return out_season_list
    def Draw_data(self):
        weight = self.down_sampling_layers[0][0].weight
        weight = weight.detach().cpu().numpy()
        np.save("/home/qihui/EXP/Mymodel_v2/view/MixingWeight/Season.weight",weight)

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """
    def __init__(self, seq_len, sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (2 ** (i + 1)),
                        seq_len // (2 ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (2 ** i),
                        seq_len // (2 ** i),
                    ),
                )
                for i in reversed(range(sampling_layers))
            ])

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        if len(trend_list) == 1:
            out_low = out_high = trend_list_reverse[0]
        else:
            out_low = trend_list_reverse[0]
            out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low.permute(0, 2, 1)).permute(0, 2, 1)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low)

        out_trend_list.reverse()
        # self.Draw_data()
        return out_trend_list
    
    
    def Draw_data(self):
        weight = self.up_sampling_layers[-1][0].weight
        weight = weight.detach().cpu().numpy()
        np.save("/home/qihui/EXP/Mymodel_v2/view/MixingWeight/Trend.weight",weight)
