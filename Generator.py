import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, lstm_shape):
        super(Generator, self).__init__()
        self.lstm = lstm_shape
        self.Lstm1 = nn.LSTM(64, lstm_shape*8, batch_first=True)
        self.later_processing = nn.Sequential(
            nn.Linear(lstm_shape * 8, lstm_shape * 2),
            nn.ReLU(),
            nn.Linear(lstm_shape * 2, lstm_shape),
            nn.ReLU(),

        )
        self.net = nn.Sequential(
            self._block(z_dim*2, features_g * 4, 4, 1, 0),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g , 4, 2, 1),
            nn.ConvTranspose2d(
                features_g , channels_img, 4, 2, 1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, prompt):
        h0 = torch.randn(1, prompt.shape[0],self.lstm*8).to("cuda")
        c0 = torch.randn(1, prompt.shape[0], self.lstm*8).to("cuda")
        prompt_result, (hn, cn) = self.Lstm1(prompt, (h0, c0))
        prompt_result = self.later_processing(prompt_result)
        prompt_result = prompt_result.view(prompt_result.size(0), -1, 1, 1)
        x = torch.cat([prompt_result,x], dim=1)
        return self.net(x)