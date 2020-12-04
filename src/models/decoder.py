from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlockInFL(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DenseBlockInFL, self).__init__()
        self.W_i = nn.Linear(input_dim, output_dim)
        self.W_f = nn.Linear(input_dim, output_dim)
        self.W_o = nn.Linear(input_dim, output_dim)
        self.W_c = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        input_i: torch.Tensor,
        input_f: torch.Tensor,
        input_o: torch.Tensor,
        input_c: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        output_i = self.W_i(input_i)
        output_f = self.W_f(input_f)
        output_o = self.W_o(input_o)
        output_c = self.W_c(input_c)
        return output_i, output_f, output_o, output_c


class FactoredLSTM(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        factored_dim: int,
        vocab_size: int,
        mode_list: List[str]
    ) -> None:
        super(FactoredLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        assert "factual" in mode_list, "mode_list must have factual"
        self.mode_list = mode_list
        # embedding
        self.B = nn.Embedding(vocab_size, emb_dim)

        # factored lstm weights
        self.U = DenseBlockInFL(factored_dim, hidden_dim)
        self.V = DenseBlockInFL(emb_dim, factored_dim)
        self.W = DenseBlockInFL(hidden_dim, hidden_dim)
        self.S_d = nn.ModuleDict({
            k: DenseBlockInFL(factored_dim, factored_dim) for k in self.mode_list
        })

        # weight for output
        self.C = nn.Linear(hidden_dim, vocab_size)

    def forward_step(
        self,
        input: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
        mode: str
    ) -> Tuple[torch.Tensor]:
        i, f, o, c = self.V(input, input, input, input)
        i, f, o, c = self.S_d[mode](i, f, o, c)
        i, f, o, c = self.U(i, f, o, c)
        h_i, h_f, h_o, h_c = self.W(h_in, h_in, h_in, h_in)

        i_t = torch.sigmoid(i + h_i)
        f_t = torch.sigmoid(f + h_f)
        o_t = torch.sigmoid(o + h_o)
        c_tilda = torch.tanh(c + h_c)

        c_t = f_t * c_in + i_t * c_tilda
        h_t = o_t * c_t

        outputs = self.C(h_t)

        return outputs, h_t, c_t

    def forward(
        self,
        captions: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        mode: str = "factual"
    ) -> torch.Tensor:
        """
        Args:
            features: fixed vectors from images, [batch, emb_dim]
            captions: [batch, max_len]
            mode: type of caption to generate
        """
        batch_size = captions.size(0)
        embedded = self.B(captions)  # [batch, max_len, emb_dim]
        # concat features and captions
        if mode == "factual":
            if image_features is None:
                raise ValueError("No image features are given")
            embedded = torch.cat((image_features.unsqueeze(1), embedded), 1)

        # initialize hidden state
        h_t = torch.Tensor(batch_size, self.hidden_dim)
        c_t = torch.Tensor(batch_size, self.hidden_dim)
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)

        all_outputs = []
        # iterate
        for ix in range(embedded.size(1) - 1):
            emb = embedded[:, ix, :]
            outputs, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
            all_outputs.append(outputs)

        all_outputs = torch.stack(all_outputs, 1)

        return all_outputs

    def beam_search(
        self,
        feature: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 30,
        mode: str = "factual",
        bos_id: int = 2,
        eos_id: int = 3
    ) -> torch.Tensor:
        """
        generate captions from feature vectors with beam search

        Args:
            features: fixed vector for an image, [1, emb_dim]
            beam_size: stock size for beam search
            max_len: max sampling length
            mode: type of caption to generate
        """
        # initialize hidden state
        h_t = torch.Tensor(1, self.hidden_dim)
        c_t = torch.Tensor(1, self.hidden_dim)
        nn.init.uniform_(h_t)
        nn.init.uniform_(c_t)

        # forward 1 step
        _, h_t, c_t = self.forward_step(feature, h_t, c_t, mode=mode)

        # candidates: [score, decoded_sequence, h_t, c_t]
        symbol_id = torch.LongTensor([1]).unsqueeze(0)
        candidates = [[0, symbol_id, h_t, c_t, [bos_id]]]

        # beam search
        t = 0
        while t < max_len - 1:
            t += 1
            tmp_candidates = []
            end_flag = True
            for score, last_id, h_t, c_t, id_seq in candidates:
                if id_seq[-1] == eos_id:
                    tmp_candidates.append([score, last_id, h_t, c_t, id_seq])
                else:
                    end_flag = False
                    emb = self.B(last_id)
                    output, h_t, c_t = self.forward_step(emb, h_t, c_t, mode=mode)
                    output = output.squeeze(0).squeeze(0)
                    # log softmax
                    output = F.log_softmax(output, dim=1)
                    output, indices = torch.sort(output, descending=True)
                    output = output[:beam_size]
                    indices = indices[:beam_size]
                    score_list = score + output
                    for score, wid in zip(score_list, indices):
                        tmp_candidates.append(
                            [score, wid, h_t, c_t, id_seq + [int(wid.data.numpy())]]
                        )
            if end_flag:
                break
            # sort by normarized log probs and pick beam_size highest candidate
            candidates = sorted(
                tmp_candidates,
                key=lambda x: -x[0].data.numpy() / len(x[-1])
            )[:beam_size]

        return candidates[0][-1]
