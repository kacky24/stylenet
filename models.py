import sys
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.module):
    def __init__(self, emb_dim):
        '''
        Load the pretrained ResNet152 and replace fc
        '''
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, emb_dim)
        # fix weights except fc
        for param in list(self.resnet.parameters())[:-1]:
            param.requires_grad = False

    def forward(self, images):
        '''Extract the image feature vectors'''
        features = self.resnet(images)
        return features


class FactoredLSTM(nn.module):
    def __init__(self, emb_dim, hidden_dim, factord_dim,  vocab_size):
        super(FactoredLSTM, self).__init__()
        # embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # factored lstm weights
        self.U_i = nn.Linear(factord_dim, hidden_dim)
        self.S_fi = nn.Linear(factord_dim, factord_dim)
        self.V_i = nn.Linear(emb_dim, factord_dim)
        self.W_i = nn.Linear(hidden_dim, hidden_dim)

        self.U_f = nn.Linear(factord_dim, hidden_dim)
        self.S_ff = nn.Linear(factord_dim, factord_dim)
        self.V_f = nn.Linear(emb_dim, factord_dim)
        self.W_f = nn.Linear(hidden_dim, hidden_dim)

        self.U_o = nn.Linear(factord_dim, hidden_dim)
        self.S_fo = nn.Linear(factord_dim, factord_dim)
        self.V_o = nn.Linear(emb_dim, factord_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.U_c = nn.Linear(factord_dim, hidden_dim)
        self.S_fc = nn.Linear(factord_dim, factord_dim)
        self.V_c = nn.Linear(emb_dim, factord_dim)
        self.W_c = nn.Linear(hidden_dim, hidden_dim)

        self.S_hi = nn.Linear(factord_dim, factord_dim)
        self.S_hf = nn.Linear(factord_dim, factord_dim)
        self.S_ho = nn.Linear(factord_dim, factord_dim)
        self.S_hc = nn.Linear(factord_dim, factord_dim)

        self.S_ri = nn.Linear(factord_dim, factord_dim)
        self.S_rf = nn.Linear(factord_dim, factord_dim)
        self.S_ro = nn.Linear(factord_dim, factord_dim)
        self.S_rc = nn.Linear(factord_dim, factord_dim)

        # weight for output
        self.C = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, h_0, c_0, mode="factual"):
        embedded = self.embedding(inputs)

        i = self.V_i(embedded)
        f = self.V_f(embedded)
        o = self.V_o(embedded)
        c = self.V_c(embedded)

        if mode == "factual":
            i = self.S_fi(i)
            f = self.S_ff(f)
            o = self.S_fo(o)
            c = self.S_fc(c)
        elif mode == "humorous":
            i = self.S_hi(i)
            f = self.S_hf(f)
            o = self.S_ho(o)
            c = self.S_hc(c)
        elif mode == "romantic":
            i = self.S_ri(i)
            f = self.S_rf(f)
            o = self.S_ro(o)
            c = self.S_rc(c)
        else:
            sys.stderr.write("mode name wrong!")

        i_t = F.sigmoid(self.U_i(i) + self.W_i(embedded))
        f_t = F.sigmoid(self.U_f(f) + self.W_f(embedded))
        o_t = F.sigmoid(self.U_o(o) + self.W_o(embedded))
        c_tilda = F.tanh(self.U_c(c) + self.W_c(embedded))

        c_t = f_t * c_0 + i_t * c_tilda
        h_t = o_t * c_t

        outputs = F.softmax(self.C(h_t))

        return outputs, h_t, c_t
