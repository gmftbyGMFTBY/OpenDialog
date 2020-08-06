from .header import *

class DualLSTM(nn.Module):
    
    def __init__(self, hidden_size, dropout=0.5):
        super(DualLSTM, self).__init__()
        self.model = nn.LSTM()
        
    def forward(self):
        pass