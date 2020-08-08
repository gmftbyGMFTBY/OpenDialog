from .header import *

'''
input:
    src: [batch]
    last_hidden: [layer, batch, hidden]
    context: [seq, batch, hidden]
output:
    output: [batch, output_size]
    next_hidden: [batch, hidden]
'''

class GRUDecoderNoAttention(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(GRUDecoderNoAttention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.rnn = nn.GRU(
                embed_size,
                hidden_size,
                num_layers=n_layers,
                dropout=(0 if n_layers == 1 else dropout))
        self.opt_layer = nn.Linear(hidden_size, output_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src_token, last_hidden):
        '''
        src_token: [batch]
        last_hidden: [1, batch, hidden]
        '''
        embed = self.embedding(src_token).unsqueeze(0)    # [1, batch, embed]
        output, hidden = self.rnn(embed, last_hidden)
        output = output.squeeze(0)    # [batch, hidden]
        
        output = self.opt_layer(output)    # [batch, output]
        return output

class GRUDecoder(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size,
                 n_layers=2, dropout=0.5):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(
                hidden_size + embed_size,
                hidden_size,
                num_layers=n_layers,
                dropout=(0 if n_layers == 1 else dropout))
        self.opt_layer = nn.Linear(hidden_size*2, output_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, last_hidden, context):
        embed = src.unsqueeze(0)    # [1, batch, embed]
        key = last_hidden.sum(axis=0)    # [batch, hidden]
        context_v = self.attention(key, context)    # [batch, hidden]
        context_v = context_v.unsqueeze(0)    # [1, batch, hidden]
        inpt = torch.cat([embed, context_v], 2)    # [1, batch, hidden+embed]
        
        output, hidden = self.rnn(inpt, last_hidden)
        output = output.squeeze(0)

        output = torch.cat([output, context_v.squeeze(0)], 1)     # [batch, hidden*2]
        output = self.opt_layer(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden
