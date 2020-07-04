import typing as t

import torch
import transformers as tfs


class MadliBert:
    """
    A MadliBert's functionality is accessed through this class so that the BERT model and tokenizer need only be
    specified once when the instance is created.

    :ivar model: A BertForMaskedLM model
    :ivar tokenizer: A BertTokenizer
    """

    def __init__(self, model: tfs.BertForMaskedLM, tokenizer: tfs.BertTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrain_name: str):
        """
        Instantiates the BERT model and tokenizer internally based on the name of that model

        `MadliBert.from_pretrained('bert-large-cased')`

        is equivalent to

        `MadliBert(BertForMaskedLM.from_pretrained('bert-large-cased'), BertTokenizer.from_pretrained('bert-large-cased'))`

        :param pretrain_name: a name of a BERT model that would be valid for calls to `from_pretrained` as implemented
        in other BERT packages
        :return: MadliBert instance
        """
        model = tfs.BertForMaskedLM.from_pretrained(pretrain_name)
        tokenizer = tfs.BertTokenizer.from_pretrained(pretrain_name)
        return cls(model, tokenizer)

    def __call__(self, text: str, k=5) -> t.List[t.Tuple[str, float]]:
        """
        Use BERT to "fill in the blank", predicting potential tokens to occupy the place of "[MASK]"
        :param text: A string bounded by the substrings "[SEP]" on either end and one instance of "[MASK]";
        "[SEP] I think I should have [MASK] for breakfast [SEP]" is a valid
        :param k: The number of results to return, defaults to 5
        :return: A sorted list of (prediction, probability) pairs for what tokens could occupy the place of [MASK]
        """
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
