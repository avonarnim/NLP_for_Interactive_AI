import torch
class TargetIdet(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        input_len,
        n_targets,
        embedding_dim
    ):
        super(TargetIdet, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_targets = n_targets
        hidden_dim = 128

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # lstm layer
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

        # linear layer
        self.fc = torch.nn.Linear(hidden_dim, n_targets)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        out = self.fc(lstm_out.view(len(x), -1)).squeeze(1)  # squeeze out the singleton length dimension that we maxpool'd over
        
        return out