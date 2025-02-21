The implementation of a Transformer Decoder Only model is described, which incorporates both Key-Value (KV) caching and absolute positional encoding. The entire code was crafted with reference to Figure 1 of the seminal paper "Attention is All You Need." However, our implementation only covers the right half of Figure 1, namely the Decoder, while preserving an interface for the Encoder within the Decoder.
![image](https://github.com/user-attachments/assets/17d0b7ba-ebe7-406b-b403-41effa9a5794)

At the conclusion of the program, the Transformer Decoder Only model is utilized for autoregressive causal sequence prediction. Furthermore, we verified whether the sequences generated using KV caching are identical to those generated without it. The experimental results indicate that when generating hundreds to thousands of tokens, the sequences produced are precisely the same.

Of equal importance, though not least, is the verification of the model's token generation speed within the program. As depicted in the following figure, KV caching can enhance the model's generation speed by up to tenfold. When the number of tokens reaches the thousands, an increase in speed by two orders of magnitude is also feasible.

![image](https://github.com/user-attachments/assets/f137a528-6a25-46e8-80e6-9972eda7b2ff)
Figure 2. The speed and sequence comparison between KV Caching and not.
