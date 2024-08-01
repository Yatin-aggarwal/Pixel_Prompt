import torch
import torch.nn as nn

class Discriminator(nn.Module):
      def __init__(self, in_channels, features, lstm_shape):
            super(Discriminator, self).__init__()
            self.lstm_shape = lstm_shape
            self.Lstm1 = nn.LSTM(64, lstm_shape*8, batch_first=True)
            self.later_processing = nn.Sequential(
                  nn.Linear(lstm_shape*8,lstm_shape*2 ),
                  nn.LeakyReLU(0.2),
                  nn.Linear(lstm_shape*2, lstm_shape*lstm_shape),
                  nn.Tanh(),
            )
            self.disc = nn.Sequential(
                  nn.Conv2d(
                        in_channels+1,
                        features,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                  ),
                  nn.BatchNorm2d(features),
                  nn.LeakyReLU(0.2),
                  self._block(features,features*2,4,2,1),
                  self._block(features*2,features*4,4,2,1),
                  nn.Conv2d(
                        features*4,
                        1,
                        kernel_size=4,
                        stride=2,
                        padding=0,
                  ),
                  nn.Sigmoid()

            )



      def _block(self, in_channels , features, kernel_size , stride , padding):
            return nn.Sequential(
                  nn.Conv2d(in_channels,
                            features,
                            kernel_size,
                            stride,
                            padding,
                            bias=False
                            ),
                  nn.BatchNorm2d(features),
                  nn.LeakyReLU(0.2),
            )

      def forward(self,x, prompt):
            h0 = torch.randn(1, prompt.shape[0], self.lstm_shape*8).to("cuda")
            c0 = torch.randn(1, prompt.shape[0], self.lstm_shape*8).to("cuda")
            prompt_result, (hn, cn) = self.Lstm1(prompt, (h0, c0))
            prompt_result = self.later_processing(prompt_result)
            prompt_result = prompt_result.view(prompt_result.size(0), 1, self.lstm_shape, self.lstm_shape)
            x = torch.cat([ x,prompt_result], dim=1)
            return self.disc(x)