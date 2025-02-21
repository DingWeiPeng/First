import copy
import math
import torch
import torch.nn as nn
# 实现多头注意力以及自注意力机制, 并且实现kv cache


def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 实现多头注意力。attention(Q, K ,V) = softmax(QK^T/sqrt(d_k))V
# query, key and value shape = (batch_size, num_heads, seq_len, d_k)
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("In attention, query shape, key shape, value shape, mask shape = ", query.shape, key.shape, value.shape, mask.shape)
    # print("In attention, scores shape = ", scores.shape)
    # 下面是最后时刻的key 和 value的shape。key shape = [batch, num_heads, seq_len, d_model//num_heads]。
    # mask shape = [1, 1, seq_len, seq_len] when masked multi head attention, mask shape = [1, 1, 1, seq_len] when multi head attention
    # In attention with kv cache, query shape, key shape, value shape, mask shape =  torch.Size([1, 8, 1, 64]) torch.Size([1, 8, 1024, 64]) torch.Size([1, 8, 1024, 64]) torch.Size([1, 1, 1024, 1024])
    # In attention with kv cache, scores shape =  torch.Size([1, 8, 1, 1024])

    if mask is not None and mask.shape[-1] == mask.shape[-2]: # 在decoder only 里，保证在Mulit head attention里不要使用掩模，此时mask不是一个方阵
        if scores.shape[2] == 1: # 这说明query是当前时刻的一个token，因此scores也是下一个tokens的scores，不需要整个mask矩阵
            mask = mask[:, :, scores.shape[-1]-1:scores.shape[-1], :scores.shape[-1]]
            # print("In attention, mask.shape = , mask = ", mask.shape, mask)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_atten = scores.softmax(dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)

    return torch.matmul(p_atten, value), p_atten


# Query, Key, Value shape = (batch_size, seq_len, d_model)
class MultiHeadedAttention(nn.Module):
    def __init__(self, batch_size, max_seq_len, h, d_model, dropout=0.1, use_kv_cache=True):
        '''
        :param batch_size: num batches
        :param max_seq_len: kv cache中预先缓存的最大序列长度所需要的空间
        :param h: num heads
        :param d_model: dimension of transformer
        :param dropout: it is usually 0.1
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # 多头自注意力机制里的投影矩阵。前三个对Q K V做投影，最后一个对concate(head1, head2,..., headk)做投影
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # 预分配缓存空间
        self.register_buffer(
            "key_cache",
            torch.zeros(batch_size, h, max_seq_len, self.d_k),
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(batch_size, h, max_seq_len, self.d_k),
        )
        self.current_seq_len = 0  # 当前缓存的长度
        self.max_seq_len = max_seq_len
        self.use_key_value_cache = use_kv_cache

    def forward(self, query, key, value, mask=None):
        "实现Attention is all you need 论文里的图2"
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        # query, key, value 的shape由(batch_size, seq_len, d_model)-->(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        # 先使用矩阵对query, key, value进行投影。该矩阵的表达式为nn.linear(d_model, d_model)

        if self.use_key_value_cache: # 取最新的query, key, value
            query, key, value = query[:, -1:, :], key[:, -1:, :], value[:, -1:, :]

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 如果启用 KV Cache，则对 key 和 value 进行缓存
        if self.use_key_value_cache:
            # 更新缓存中的 key 和 value
            seq_len = key.size(2)  # 当前 key 的长度
            self.key_cache[:, :, self.current_seq_len:self.current_seq_len + seq_len, :] = key
            self.value_cache[:, :, self.current_seq_len:self.current_seq_len + seq_len, :] = value

            # 更新当前缓存的长度
            self.current_seq_len += seq_len
            assert self.current_seq_len <= self.max_seq_len
            # 使用缓存中的 key 和 value
            key = self.key_cache[:, :, :self.current_seq_len, :]
            value = self.value_cache[:, :, :self.current_seq_len, :]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k))

        # 可以删除 kye value
        del query
        del key
        del value
        # 使用最后一个矩阵，对多个头得到的注意力矩阵进行投影
        return self.linears[-1](x)


# Layer Normalization 通过对输入数据进行均值和标准差的归一化处理，改变了数据的尺度和位置，但不会改变数据的分布形状。
# 这种操作有助于加速模型的训练过程，并提高模型的稳定性和性能。
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # 参数传递：通过调用 super()，可以确保任何传递给 子类LayerNorm 的参数（例如 features 和 eps）
        # 都能被正确地传递给父类 nn.Module 的构造函数。
        # features是输入数据的特征的维度大小，通常是最后一个维度的大小。
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 用到了广播机制
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    " A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last"
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x1 = self.dropout(sublayer(self.norm(x)))
        # print("In SublayerConnection, x1 shape, x shape = ", x1.shape, x.shape)
        x1_seq_len = x1.shape[1]
        return x[:, -x1_seq_len:, :] + x1


class SublayerConnection_QKV(nn.Module):
    " A residual connection followed by a layer norm. Note for code simplicity the norm is first as opposed to last"
    def __init__(self, size, dropout):
        super(SublayerConnection_QKV, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(q + self.dropout(sublayer(q, k, v)))


class Self_Attention(nn.Module):
    def __init__(self, h, d_model, size, dropout):
        '''
        :param h: 注意力机制的头数
        :param d_model: 模型的宽度，可以理解为输入数据的宽度或者单个embedding的长度
        :param size: LayerNorm中层正则化中数据的最后一个维度的大小。也就是经过注意力计算后，数据的特征维度大小，即最后一个维度的大小
        :param dropout: dropout的比例
        '''
        super(Self_Attention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.size = size
        self.dropout = dropout

        self.self_attention = MultiHeadedAttention(self.h, self.d_model, self.dropout)
        self.sublayer_connection = SublayerConnection(self.size, self.dropout)

    def forward(self, x):
        # 使用lambda实现匿名函数。这意味着SublayerConnection类里面的sublayer是self.self_attention，
        # 并且self.norm会对sublayer里面的每一个输入参数都做LayerNorm。这也实现了参数数目的可拓展性
        return self.sublayer_connection(x, lambda x: self.self_attention(x, x, x))


class Q_K_V_Attention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(Q_K_V_Attention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = dropout

        self.q_k_v_attention = MultiHeadedAttention(self.h, self.d_model, self.dropout)
        self.sublayer_connection = SublayerConnection_QKV(self.d_model, self.dropout)

    def forward(self, q, k, v):
        return self.sublayer_connection(q, k, v, self.q_k_v_attention)
