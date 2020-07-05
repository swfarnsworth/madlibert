# MadliBERT
MadliBERT uses pretrained BERT models to predict what token is most likely to occupy the place
of a missing token in a given sample.

```python
>>> from madlibert import MadliBert
>>> mb = MadliBert.from_pretrained('bert-large-cased')
>>> mb('[SEP] We should go to [MASK] today. [SEP]')
[('school', 0.2814341187477112), 
 ('class', 0.12625090777873993), 
 ('lunch', 0.0666893795132637), 
 ('it', 0.044688645750284195), 
 ('work', 0.03584672883152962)]
```

The values returned are a sorted list of prediction-probability pairs; the `k`
parameter can be set to any positive integer, defaulting to `5`.

This package uses [Transformers](https://github.com/huggingface/transformers) and
[PyTorch](https://github.com/pytorch/pytorch) to access BERT models and perform
calculations. These calculations can be performed on a CUDA-enabled device for
faster performance. Any BERT model compatible with these packages can be used.

## Installation
```bash
pip install git+https://github.com/swfarnsworth/madlibert.git
```
