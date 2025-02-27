import torch
import torch.nn as nn

class ZeroConvNet(nn.Module):
    def __init__(self, unet, in_channels=13, out_channels=4):
        super(ZeroConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=1),
            nn.Conv2d(12, 8, kernel_size=1),
            nn.Conv2d(8, 6, kernel_size=1),
            nn.Conv2d(6, out_channels, kernel_size=1)
        )
        # self._identity_initialize_weights()
        
        self.unet = unet

    def _uniform_initialize_weights(self):
        """ Uniform Scaling (Stable Diffusion Style) inspired by OpenAI """
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                scale = 1.0 / (layer.in_channels ** 0.5)
                nn.init.uniform_(layer.weight, -scale, scale)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def _zero_initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.zeros_(layer.weight)  # Initialize weights to zero
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def _identity_initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                with torch.no_grad():
                    layer.weight.zero_()  # Start with zeros
                    identity_size = min(layer.in_channels, layer.out_channels)
                    for i in range(identity_size):
                        layer.weight[i, i, 0, 0] = 1  # Set identity values
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        outputs = self.net(noisy_latents)
        outputs = self.unet(outputs, timesteps, encoder_hidden_states=encoder_hidden_states)
        return outputs
