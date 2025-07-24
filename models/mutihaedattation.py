import torch.nn as nn

class Attation_Cov(nn.Module):
    def __init__(self, d_model,  seq_len ,inner_div = 512, layer = 1):
        super().__init__()
        # self.liner2transformer1o = nn.Linear(d_model, inner_div)
        # self.liner2transformer1s = nn.Linear(d_model, inner_div)
        # self.liner2transformer1t = nn.Linear(d_model, inner_div)
        self.tansformer_o = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=inner_div, num_heads=8, dropout=0.05) for i in range(layer)]
        )
        # self.tansformer_s = nn.ModuleList(
        #     [nn.MultiheadAttention(embed_dim=inner_div, num_heads=8, dropout=0.05) for i in range(layer)]
        # )
        # self.tansformer_t = nn.ModuleList(
        #     [nn.MultiheadAttention(embed_dim=inner_div, num_heads=8, dropout=0.05) for i in range(layer)]
        # )
        
        self.transformer2liner_o = nn.Linear(inner_div,d_model)
        # self.transformer2liner_s = nn.Linear(inner_div,d_model)
        # self.transformer2liner_t = nn.Linear(inner_div,d_model)

        self.atta_cov_o = nn.Conv1d(seq_len, seq_len, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.atta_cov_s = nn.Conv1d(seq_len, seq_len, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        self.atta_cov_t = nn.Conv1d(seq_len, seq_len, kernel_size=3, stride=1, padding=1, padding_mode='circular')



    def forward(self, o):
        # o_in = self.liner2transformer1o(o)
        # s_in = self.liner2transformer1s(s)
        # t_in = self.liner2transformer1t(t)

        for lay in self.tansformer_o:
            o= lay(o, o, o)[0]+o

        # for lay in self.tansformer_s:
        #     s_in = lay(s_in, s_in, s_in)[0]
        # for lay in self.tansformer_t:
        #     t_in = lay(t_in, t_in, t_in)[0]

        o = self.atta_cov_o(self.transformer2liner_o(o))
        # s_in = self.atta_cov_s(self.transformer2liner_s(s_in))
        # t_in = self.atta_cov_t(self.transformer2liner_t(t_in))
        # return o_in,s_in,t_in
        return o