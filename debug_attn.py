import torch

B, T, C = 2, 4, 3       # toy sizes
x = torch.randn(B, T, C)

#attn = CausalSelfAttention(GPTConfig)  # 4 heads
#y = attn(x)
print(x.shape)            # should be (2, 4, 32)
print(x.stride())
print(x.reshape((1, 4,2,3)).shape)

print(x.view((1,4,2,3)).shape)

y = x.permute(1,0,2)
print(y.shape)
print(y.reshape(-1).shape)

y_c = y.contiguous()
print(y_c.shape)
y_s = y_c.stride()
print(y_s)