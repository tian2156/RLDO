import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention, Mlp


class Block(
    nn.Module
):  
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()  
       
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       
        self.atn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
       
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
       
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
       
        approx_gelu = lambda: nn.GELU(approximate="tanh")
     

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        

    def forward(
        self, x
    ):  
        x = (
            self.atn(self.norm1(x)) + x
        )  
        x = (
            self.mlp(self.norm2(x)) + x
        ) 
        return x


class Model(nn.Module):
    def __init__(
        self, n, depth, hidden_size=512, num_heads=8, mlp_ratio=4.0, **block_kwargs
    ):
        
        super().__init__()

        self.n = n
        
        self.topo_emb = nn.Linear(
            n, hidden_size
        ) 
        self.weight_emb = nn.Linear(
            1, hidden_size
        )  
        self.n_emb = nn.Embedding(
            10, hidden_size
        )  

        self.blocks = nn.Sequential(  
            *[  
                Block(
                    hidden_size, num_heads, mlp_ratio, **block_kwargs
                )  
                for _ in range(
                    depth
                )  
            ]  
        )  
       

        self.output = nn.Linear(hidden_size, 2 * n)  
    def forward(
        self, topo, weight
    ):  

        t = self.topo_emb(
            topo.float() + topo.transpose(1, 2).float()
        )  

        w = self.weight_emb(
            weight.float().reshape(-1, self.n, 1)
        )  
        n = self.n_emb(
            torch.arange(self.n, device=topo.device)
            .unsqueeze(0)
            .repeat(topo.shape[0], 1)
        )

        x = t + w + n  
        x = self.blocks(x)
        x = self.output(x)
        x = x.reshape(-1, self.n, self.n, 2)
        x = (x + x.transpose(1, 2)) / 2
        x[:, :, :, 0] = x[:, :, :, 0].masked_fill(topo == 0, -1e9)
        x = nn.functional.softmax(x, dim=-1)
       

        return x
