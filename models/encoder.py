from .header import *

'''
Encoder component for ReGe
inpt: 
    src: [seq, batch, embedding]
    src_l: [batch]
opt:
    context: [seq, batch, hidden]
    hidden: [batch, hidden]
'''

class GRUEncoder(nn.Module):

    def __init__(self, embed_size, hidden_size, 
                 n_layers=1, dropout=0.5, bidirectional=True):
        super(GRUEncoder, self).__init__()
        self.bidirectional = bidirectional
        self.n_layer = n_layers
        self.hidden_size = hidden_size
        # self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(
                embed_size, 
                hidden_size, 
                num_layers=n_layers, 
                dropout=(0 if n_layers == 1 else dropout),
                bidirectional=bidirectional)
        self.times = n_layers*2 if bidirectional else n_layers
        self.hidden_project = nn.Linear(self.times*hidden_size, hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, src_l):
        # src: [seq, batch, embed]
        embed = nn.utils.rnn.pack_padded_sequence(src, src_l, enforce_sorted=False)
        output, hidden = self.rnn(embed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # output: [seq, batch, hidden * bidirectional]
        # hidden: [n_layer * bidirectional, batch, hidden]
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden.permute(1, 2, 0)    # [batch, hidden, n_layer * bidirectional]
        hidden = hidden.reshape(hidden.shape[0], -1)    # [batch, hidden*...]
        hidden = torch.tanh(self.hidden_project(hidden))     # [batch, hidden]
        return output, hidden
