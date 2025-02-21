# 实现Transformer Decoder Only with KV Cache。从attention文件里面就包括了Transformer Decoder Only所有的成分
# Transformer Decoder Only includes Positional Encoding,
# Masked Multi-Head Attention, Add&Norm,
# Multi-Head Attention, Add&Norm
# Feed Forward, Add&Norm
# Realized by Dingwei peng, whose email is jackdawsonabc@163.com in 02/11.2025. Welcome to concate me.
from attention import *


class PositionalEncoding(nn.Module):
    '''
    Implement the Positonal Encoding function with max length.
    x shape is (batch_size, seq_len, d_model). seq_len will grow as the time
    pe shape is (1, max_len, d_model)
    '''

    def __init__(self, d_model, dropout, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10_000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print("In PositionalEncoding, x shape, self.pe shape = ", x.shape, self.pe.shape)
        "这里应该选择pe的前若干个位置与x相加。x.size(1)代表了x的序列长度"
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Decoder(nn.Module):
    "Generic N layer decoder with masking"

    def __init__(self, layer, N, d_model, dropout, tgt_embed):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.tgt_embed = tgt_embed
        self.position_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.generator = Generator

    def forward(self, x, src_mask, tgt_mask, memory=None):
        '''
        :param x: shape is (n_batches, seq_len). After self.tgt_embed(x), shape is (batch_size, seq_len, d_model)
        :param memory: memory can come from transformer encoder and decoder.
        :param src_mask: src comes from transformer encoder, so src_mask don't need to be.
        :param tgt_mask: tgt_mask is used in Masked Multi-Head Attention for causality
        :return: the data which is processed by N Masked Multi-Head Attention and N Multi-Head Attention Layers.
        '''
        x = self.tgt_embed(x)
        x = self.position_encoding(x)
        for layer in self.layers:
            # print("In Decoder, before one decoder layer, x shape = ", x.shape)
            x = layer(x=x, src_mask=src_mask, tgt_mask=tgt_mask, memory=memory)
            # print("In Decoder, after one decoder layer, x shape = ", x.shape)
        return self.norm(x)


class DecoderLayer(nn.Module):
    '''
    Decoder is made of self-attn, src-attn, and feed forward (defined below).
    Decoder provides the query in src_atten, while encoder offers key and value.
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, src_mask, tgt_mask, memory=None):
        "Follow Figure 1 (right) for connections."
        # print("In DecoderLayer, x shape = ", x.shape)
        m = x if memory is None else memory
        x = self.sublayer[0](x, lambda x: self.self_attn(query=x, key=x, value=x, mask=tgt_mask))
        # print("In DecoderLayer, x shape = ", x.shape)
        x = self.sublayer[1](x, lambda x: self.src_attn(query=x, key=m, value=m, mask=src_mask))
        # print("In DecoderLayer, x shape = ", x.shape)
        return self.sublayer[2](x, self.feed_forward)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.functional.log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def subsequent_mask(size):
    "Mask out subsequent positions"
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def make_model(max_seq_len, num_vocbs=1024, N=1, d_model=512, d_ff=1024, h=8, dropout=0.1, use_cache=True):
    '''
    :param num_vocbs: 码本大小
    :param max_seq_len: 序列的最大长度
    :param N: Decoder Only的层数，通过copy.deepcopy拷贝得到
    :param d_model: 模型宽度
    :param d_ff: 前馈传播里的模型宽度
    :param h: 注意力的头数
    :param dropout: 通常为0.1
    :param use_cache: 使用cache
    :return: 返回值
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(batch_size=1, max_seq_len=max_seq_len, h=h, d_model=d_model, use_kv_cache=use_cache)
    ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
    position = PositionalEncoding(d_model=d_model, dropout=dropout)
    tgt_embed = Embeddings(vocab=num_vocbs, d_model=d_model)
    # model并未携带最后的Linear和Softmax层网络
    # layer, N, d_model, dropout, tgt_embed
    model = Decoder(layer=DecoderLayer(d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=ff, dropout=dropout),
                    N=N, d_model=d_model, dropout=dropout, tgt_embed=tgt_embed)
    generator = Generator(d_model, num_vocbs)

    for p in generator.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return model, generator, position


def inference_test():
    import time
    use_cache = True
    assert use_cache is True # 调试程序设定先使用KV CACHE，然后不使用KV CACHE

    num_vocbs = 1024 # 码本大小
    decoder_only_model, generator, position = make_model(max_seq_len=1024, num_vocbs=num_vocbs, N=4, use_cache=use_cache)
    # eval模式能够保证模型里的Dropout不被使用，这是KV_CACHE和没有KV_CACHE时模型预测结果一致的保证
    decoder_only_model.eval()
    generator.eval()
    position.eval()

    src = torch.LongTensor([[1]])
    # 这里保留src_mask仅仅是为了程序能够拓展到Encoder-Decoder架构。实际上，src_mask并不能起到任何作用，因为它所有的值是1.
    ys_kv_cache = torch.zeros(1, 1).type_as(src)

    # tgt_mask用于attention.py里attention函数，保证序列的因果预测
    # x shape = (n_batches, seq_len)，经过Output Eebedding 和 Positional Encoding 后，shape 变为(n_batches, seq_len, d_model)
    # 在经过Masked Multi-Head Attention 或者 Multi-Head Attention后，
    # 当不使用KV_Cache时，注意力矩阵的shape都是(n_batches, num_heads, seq_len, seq_len)
    tokens = 512  # 因果预测tokens的个数为tokens - 1
    assert tokens <= num_vocbs # num_vocbs代表码本大小，所以码本的索引是[0, num_vocbs-1]。为了保证index在这个范围，tokens <= num_vocbs

    start_time = time.time()
    for index in range(1, tokens): # 已知src，推测下一个元素
        src = torch.LongTensor([[index2 for index2 in range(1, index+1)]])
        # src_mask仅仅是为了程序能够拓展到语言翻译里面的Encoder-Decoder架构。实际上，src_mask要求等于tgt_mask，在序列因果预测里.
        # src_mask = torch.ones(1, 1, index)
        tgt_mask = subsequent_mask(index).type_as(src.data)
        out = decoder_only_model(
            x=src, src_mask=tgt_mask, tgt_mask=tgt_mask, memory=None
        )
        prob = generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys_kv_cache = torch.cat([ys_kv_cache, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print("Example Untrained Model Prediction with kv cache\n", ys_kv_cache)
    print("*" * 20)
    key_value_cache_time = time.time()-start_time

    start_time = time.time()
    for layer in decoder_only_model.layers:
        layer.self_attn.use_key_value_cache = 1 - use_cache
        layer.src_attn.use_key_value_cache = 1 - use_cache

    ys_no_kv_cache = torch.zeros(1, 1).type_as(src)
    for index in range(1, tokens): # 已知src，推测下一个元素
        src = torch.LongTensor([[index2 for index2 in range(1, index+1)]])
        # 这里保留src_mask仅仅是为了程序能够拓展到Encoder-Decoder架构。实际上，src_mask并不能起到任何作用，因为它所有的值是1.
        # src_mask = torch.ones(1, 1, index)
        tgt_mask = subsequent_mask(src.shape[-1]).type_as(src.data)
        out = decoder_only_model(
            x=src, src_mask=tgt_mask, tgt_mask=tgt_mask, memory=None
        )
        prob = generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys_no_kv_cache = torch.cat([ys_no_kv_cache, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    print("Example Untrained Model Prediction w/o kv cache\n", ys_no_kv_cache)
    print("*" * 20)
    no_key_value_cache_time = time.time() - start_time

    compare_result = torch.equal(ys_kv_cache, ys_no_kv_cache)
    if compare_result:
        print("In autoregressive causal prediction, "
              "the prediction result is EXACTLY equal whether there is key value cache or not.")
    else:
        print("In autoregressive causal prediction, "
              "the prediction result from key value cache or not is NOT equal ")
    print(f"Without KV Cache: {no_key_value_cache_time:.4f}s, while generating {tokens - 1} tokens.")
    print(f"With KV Cache: {key_value_cache_time:.4f}s, while generating {tokens - 1} tokens.")


def run_tests():
    with torch.no_grad():
        inference_test()


# 至于怎么实现KV_Cache元素，那就需要
if __name__ == "__main__":
    run_tests()
