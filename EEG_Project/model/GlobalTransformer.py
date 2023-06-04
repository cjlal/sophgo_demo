import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer_GA(nn.Module):
    def __init__(self, num_nodes, dim_model, num_head, att_dim, hidden, dropout, num_encoder):
        super(Transformer_GA, self).__init__()
        self.model_name = 'Transformer'
        self.encoder = Encoder_GA(num_nodes, dim_model, num_head, att_dim, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_encoder)])  # 使用两个Encoder，尝试6个encoder发现存在过拟合，毕竟数据集量比较少（10000左右），可能性能还是比不过LSTM

    def forward(self, x):
        out = x
        for encoder in self.encoders:
            out = encoder(out)
        return out


class Encoder_GA(nn.Module):
    def __init__(self, num_nodes, dim_model, num_head, att_dim, hidden, dropout):
        super(Encoder_GA, self).__init__()
        self.global_attention = MultiHead_Global_Attention(num_nodes=num_nodes, in_dim=dim_model, num_heads=num_head,
                                                         att_dim=att_dim)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.global_attention(x)
        out = self.feed_forward(out)
        return out


class MultiHead_Global_Attention(nn.Module):
    def __init__(self, num_nodes, in_dim, num_heads=8, att_dim=64, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):
        super(MultiHead_Global_Attention, self).__init__()
        self.num_heads = num_heads
        self.out_dim = att_dim
        assert self.out_dim % num_heads == 0

        self.trans_dims_q = nn.Linear(in_dim, self.out_dim)  # q
        self.trans_dims_k = nn.Linear(in_dim, self.out_dim)  # k
        self.trans_dims_v = nn.Linear(in_dim, self.out_dim)  # v

        self.k = num_nodes
        self.linear_0 = nn.Linear(self.out_dim // self.num_heads, self.k, bias=qkv_bias)  # M_k

        self.linear_1 = nn.Linear(self.k, self.out_dim // self.num_heads, bias=qkv_bias)  # M_v

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(self.out_dim, in_dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Query
        Q = self.trans_dims_q(x)  # B, N, C -> B, N, out_dim
        Q = Q.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_head, N, out_dim//num_head
        # Key
        K = self.trans_dims_k(x)  # B, N, C -> B, N, out_dim
        K = K.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_head, N, out_dim//num_head
        # Value
        V = self.trans_dims_v(x)  # B, N, C -> B, N, out_dim
        V = V.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_head, N, out_dim//num_head

        # Q*M_k*M_v
        attn1 = self.linear_0(Q)  # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn1 = attn1.softmax(dim=-2)
        attn1 = attn1 / (1e-9 + attn1.sum(dim=-1, keepdim=True))
        attn1 = self.attn_drop(attn1)
        out1 = self.linear_1(attn1).permute(0, 2, 1, 3).reshape(B, N, -1)  # B, num_head, N, N -> B, N, out_dim
        # Q*K*M_v
        attn2 = torch.matmul(Q, K.transpose(2, 3))  # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn2 = attn2.softmax(dim=-2)
        attn2 = attn2 / (1e-9 + attn2.sum(dim=-1, keepdim=True))
        attn2 = self.attn_drop(attn2)
        out2 = self.linear_1(attn2).permute(0, 2, 1, 3).reshape(B, N, -1)  # B, num_head, N, N -> B, N, out_dim
        # Q*M_k*V
        attn3 = self.linear_0(Q)  # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn3 = attn3.softmax(dim=-2)
        attn3 = attn3 / (1e-9 + attn3.sum(dim=-1, keepdim=True))
        attn3 = self.attn_drop(attn3)
        out3 = torch.matmul(attn3, V).permute(0, 2, 1, 3).reshape(B, N, -1)  # B, num_head, N, N -> B, N, out_dim

        out = out1 + out2 + out3
        out = self.proj(out)  # B, N, out_dim -> B, N, C
        out = self.proj_drop(out)
        x = x + out
        x = F.relu(x)
        return x


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)  # (64,64)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class GlobalTransformer(nn.Module):
    """
    Multi-branch dynamic graph convolution global Transformer network.
    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, num_T_head, att_dim, hidden, num_encoder, graph_size):
        super(GlobalTransformer, self).__init__()
        self.transformer1 = Transformer_GA(num_nodes=graph_size, dim_model=in_c, num_head=num_T_head, att_dim=att_dim,
                                           hidden=hidden,
                                           dropout=0.5, num_encoder=num_encoder)
        self.fc1 = nn.Linear(graph_size*in_c, graph_size * in_c * 2)  # 62*5 (4278)  27*64(1863)  23(1587)
        self.classifier = nn.Linear(graph_size * in_c * 2, 3)

    def forward(self, inputs):
        B = inputs.size(0)
        out = self.transformer1(inputs)
        out = out.view(B,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    node = 62
    x = torch.randn(32, 62, 5).cuda(0)
    net = GlobalTransformer(in_c = 5, num_T_head = 8, att_dim = 64, hidden = 512, num_encoder = 4, graph_size = 62).cuda(0)
    out = net(x)
    print(out)