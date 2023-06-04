import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConvolution,Linear
from model.utils import normalize_A
from math import sqrt

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out, dropout,device="cpu"):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out,device))

    def forward(self, x,L):
        #  generate_cheby_adj(L, self.K)
        support = []
        for i in range(self.K):
            if i == 0:
                support.append(torch.eye(L.shape[1]).to(L.device))
            elif i == 1:
                support.append(L)
            else:
                temp = torch.matmul(support[-1], L)
                support.append(temp)
        adj = support
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3,dropout = 0.5,device = "cpu"):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers
        #num_out: num_features_out
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out, dropout,device)
        self.BN1 = nn.BatchNorm1d(xdim[2]).cuda()
        self.fc1 = Linear(xdim[1] * num_out, 64)
        self.fc2=Linear(64,nclass)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).to(device))
        nn.init.xavier_normal_(self.A)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result=self.fc2(result)
        return result
class GCBnet(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3,dropout=0.5,device = "cpu"):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers
        #num_out: num_features_out
        super(GCBnet, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out, dropout,device)
        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.fc1 = Linear(xdim[1] * 88, 3)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).to(device))
        nn.init.xavier_normal_(self.A)

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=xdim[1], out_channels=xdim[1], kernel_size=7, stride=2, padding=3),
        #     nn.ReLU(),
        #     )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2,stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
    def forward(self, x):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        # L = normalize_A(self.A)
        # result = self.layer1(x, L)
        # result = self.conv1(result)
        # result = torch.transpose(result,1,2)
        # result = self.maxpool(result)
        # result = torch.transpose(result, 1, 2)
        # result = self.conv2(result)
        # result = result.reshape(x.shape[0], -1)
        # result = F.relu(self.fc1(result))
        # result = self.fc2(result)
        # return result
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        gcnn = self.layer1(x, L)
        result = torch.transpose(gcnn, 1, 2)
        conv1 = self.conv1(result)
        maxpool = self.maxpool(conv1)
        conv2 = self.conv2(maxpool)
        out1 = gcnn.reshape(x.shape[0],-1,1)
        out2 = maxpool.reshape(x.shape[0],-1,1)
        out3 = conv2.reshape(x.shape[0],-1,1)
        result = torch.cat([out1,out2,out3],1)

        result = result.reshape(x.shape[0], -1)
        # result = F.relu(self.fc1(result))
        result = self.fc1(result)
        return result

class GCNHyperAttention(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3,dropout=0.5):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers
        #num_out: num_features_out
        super(GCNHyperAttention, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.fc1 = Linear(xdim[1] * num_out, 64)
        self.fc2=Linear(64,nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A)
        self.mha = MultiHead_Hyper_Attention(num_nodes=xdim[1], in_dim=num_out, num_heads=8, att_dim=64)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = self.mha(result)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result=self.fc2(result)
        return result

class GCNCFAttention(nn.Module):
    def __init__(self, xdim, k_adj, num_out, nclass=3,dropout=0.5):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers
        #num_out: num_features_out
        super(GCNCFAttention, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out, dropout)
        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.fc1 = Linear(xdim[1] * num_out, 64)
        self.fc2=Linear(64,nclass)
        # self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A)
        self.cf_attention = MultiHead_CFAttention(dim_in=num_out, dim_k=64, dim_v=64, num_heads=4)

    def forward(self, x):
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = self.cf_attention(result)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result=self.fc2(result)
        return result

class MultiHead_CFAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=4):
        super(MultiHead_CFAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0 # "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in//2
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

        self.fc = nn.Linear(dim_k*2, dim_in)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(dim_in)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in * 2

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head
        # Channel Attention
        q_c = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k_c = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v_c = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist_c = torch.matmul(q_c, k_c.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        # dist = dist.masked_fill_(dist==0,-np.inf)
        dist_c = torch.softmax(dist_c, dim=-1)  # batch, nh, n, n

        att_c = torch.matmul(dist_c, v_c)  # batch, nh, n, dv
        att_c = att_c.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        # Feature Attention
        q_f = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k_f = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v_f = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist_f = torch.matmul( k_f.transpose(2, 3),q_f) * self._norm_fact  # batch, nh, dk, dk
        # dist = dist.masked_fill_(dist==0,-np.inf)
        dist_f = torch.softmax(dist_f, dim=-1)  # batch, nh, dk, dk

        att_f = torch.matmul(v_f , dist_f)  # batch, nh, n, dv
        att_f = att_f.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        att = torch.cat([att_c,att_f],dim=2)
        out = self.fc(att)# todo add a relu
        out = self.act(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out

class MultiHead_Hyper_Attention(nn.Module):
    def __init__(self, num_nodes,in_dim,num_heads=8,  att_dim = 64, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MultiHead_Hyper_Attention,self).__init__()
        self.num_heads = num_heads
        self.out_dim = att_dim
        assert self.out_dim % num_heads == 0

        self.trans_dims_q = nn.Linear(in_dim, self.out_dim)  # q
        self.trans_dims_k = nn.Linear(in_dim, self.out_dim)  # k
        self.trans_dims_v = nn.Linear(in_dim, self.out_dim)  # v

        self.k = num_nodes
        self.linear_0 = nn.Linear( self.out_dim// self.num_heads, self.k,bias=qkv_bias) # M_k

        self.linear_1 = nn.Linear(self.k, self.out_dim // self.num_heads,bias=qkv_bias) # M_v

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(self.out_dim, in_dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Query
        Q = self.trans_dims_q(x)  # B, N, C -> B, N, out_dim
        Q = Q.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3) # B, num_head, N, out_dim//num_head
        # Key
        K = self.trans_dims_k(x)  # B, N, C -> B, N, out_dim
        K = K.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_head, N, out_dim//num_head
        # Value
        V = self.trans_dims_v(x)  # B, N, C -> B, N, out_dim
        V = V.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B, num_head, N, out_dim//num_head


        # Q*M_k*M_v
        attn1 = self.linear_0(Q) # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn1 = attn1.softmax(dim=-2)
        attn1 = attn1 / (1e-9 + attn1.sum(dim=-1, keepdim=True))
        attn1 = self.attn_drop(attn1)
        out1 = self.linear_1(attn1).permute(0, 2, 1, 3).reshape(B, N, -1)# B, num_head, N, N -> B, N, out_dim
        # Q*K*M_v
        attn2 = torch.matmul(Q,K.transpose(2,3))  # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn2 = attn2.softmax(dim=-2)
        attn2 = attn2 / (1e-9 + attn2.sum(dim=-1, keepdim=True))
        attn2 = self.attn_drop(attn2)
        out2 = self.linear_1(attn2).permute(0, 2, 1, 3).reshape(B, N, -1)  # B, num_head, N, N -> B, N, out_dim
        # Q*M_k*V
        attn3 = self.linear_0(Q)  # B, num_head, N, out_dim//num_head -> B, num_head, N, N
        attn3 = attn3.softmax(dim=-2)
        attn3 = attn3 / (1e-9 + attn3.sum(dim=-1, keepdim=True))
        attn3 = self.attn_drop(attn3)
        out3 = torch.matmul(attn3,V).permute(0, 2, 1, 3).reshape(B, N, -1)  # B, num_head, N, N -> B, N, out_dim

        out = out1 + out2 + out3
        out = self.proj(out) #B, N, out_dim -> B, N, C
        out = self.proj_drop(out)
        x = x + out
        x = F.relu(x)
        return x

if __name__ == "__main__":
    # net = ChebNet(in_c=3, hid_c=6,out_c=2, K=2)
    print(torch.__version__)
    x = torch.randn(16, 62, 5).cuda(0)
    xdim = x.shape
    # net = GCNHyperAttention(xdim=xdim,k_adj=5,num_out=64,nclass=3).cuda(0)
    net = GCBnet(xdim=xdim, k_adj=5, num_out=64, nclass=3, dropout=0.5).cuda(0)
    # net = GCNCFAttention(xdim=xdim,k_adj=5,num_out=64,nclass=3).cuda(0)
    # net = DGCNN(xdim=xdim,k_adj = 5,num_out=64,nclass=3).cuda(0)
    # GPU
    # net.cuda(0)
    # GPU
    # x = torch.randn(32, 62, 5).cuda(0)
    # CPU
    out = net(x)
    print(out)
