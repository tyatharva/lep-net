# vision_transformer.py
import torch
import torch.nn as nn

class ViTGenerator(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, out_channels=3, embed_dim=768, num_heads=8, num_layers=12):
        super(ViTGenerator, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, activation='gelu')
            for _ in range(num_layers)
        ])

        # Decoder to map embeddings back to image patches
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * out_channels),
            nn.Tanh()  # Assuming output is normalized to [-1, 1]
        )

    def forward(self, x):
        # Flatten input image into patches
        patches = self.patch_embed(x)  # Shape: [batch_size, embed_dim, H/patch_size, W/patch_size]
        batch_size, embed_dim, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]

        # Add position embeddings
        patches = patches + self.position_embeddings

        # Pass through transformer layers
        for layer in self.transformer_layers:
            patches = layer(patches)

        # Decode each patch back to image pixels
        output = self.decoder(patches)  # Shape: [batch_size, num_patches, patch_size * patch_size * out_channels]
        output = output.view(batch_size, H, W, self.patch_size, self.patch_size, self.out_channels)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()  # Rearrange to [batch, out_channels, H*patch_size, W*patch_size]
        output = output.view(batch_size, self.out_channels, self.img_size, self.img_size)  # Final shape

        return output
