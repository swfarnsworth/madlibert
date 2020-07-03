import os
import typing as t

import torch
import transformers as tfs

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


class MadliBert:

    def __init__(self, model: tfs.BertForMaskedLM, tokenizer: tfs.BertTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrain_name: str):
        model = tfs.BertForMaskedLM.from_pretrained(pretrain_name)
        tokenizer = tfs.BertTokenizer.from_pretrained(pretrain_name)
        return cls(model, tokenizer)

    def __call__(self, text, k=5) -> t.List[t.Tuple[str, float]]:
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        token_tensor = torch.tensor(indexed_tokens)
        mask_tensor = torch.tensor([token != '[MASK]' for token in tokenized_text], dtype=torch.float)

        with torch.no_grad():
            result = self.model(token_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), masked_lm_labels=None)

        result = result[0].squeeze(0)
        scores = torch.softmax(result, dim=-1)

        mask_index = tokenized_text.index('[MASK]')
        topk_prob, topk_indices = torch.topk(scores[mask_index, :], k, sorted=True)
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_indices)

        return [(t, float(p)) for t, p in zip(topk_tokens, topk_prob)]
