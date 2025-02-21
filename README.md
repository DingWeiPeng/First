The implementation of a Transformer Decoder Only model is described, which incorporates both Key-Value (KV) caching and absolute positional encoding. The entire code was crafted with reference to Figure 1 of the seminal paper "Attention is All You Need." However, our implementation only covers the right half of Figure 1, namely the Decoder, while preserving an interface for the Encoder within the Decoder.

<p align="center">
  <img src="https://github.com/user-attachments/assets/17d0b7ba-ebe7-406b-b403-41effa9a5794" alt="image">
  <br>
</p>

At the conclusion of the program, the Transformer Decoder Only model is utilized for autoregressive causal sequence prediction. Furthermore, we verified whether the sequences generated using KV caching are identical to those generated without it. The experimental results indicate that when generating hundreds to thousands of tokens, the sequences produced are precisely the same.

The program was written following the principle of low cohesion and high coupling, where each module in the Decoder is implemented as a class. These classes call each other without exhibiting any obvious containment relationships. Consequently, when first reading KV_Cache_Transformer.py and attention.py, you may find it challenging to grasp the program's logic and appreciate its sophistication. If you find the program confusing, please review it multiple times and try debugging it. 

The execution of the KV_Cache_Transformer.py requires only the presence of PyTorch. It is essential to ensure that the number of tokens does not exceed the number of vocabulary entries (tokens ≤ num_vocbs).

Of equal importance, though not least, is the verification of the model's token generation speed within the program. Figure 2 illustrates that when generating 511 tokens in a causal autoregressive way, the sequences produced with and without KV caching are EXACTLY IDENTICAL. It is particularly noteworthy that KV caching enhances the generation speed by a factor of ten on the CPU. From Figure 2 and Figure 3,  an increase in speed by one orders of magnitude is also feasible When the number of tokens reaches the thousands.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f137a528-6a25-46e8-80e6-9972eda7b2ff" alt="image">
  <br>
  Figure 2. The speed and sequence comparison between KV Caching and not in CPU while generating 511 tokens.
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/436b6b9f-2832-4f7a-a4a3-d497d0c6ac3f" alt="image">
  <br>
  Figure 3. The speed and sequence comparison between KV Caching and not in CPU while generating 1023 tokens.
</p>
