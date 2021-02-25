# Next frame embedding prediction

The dynamics training code here is based off of the [word-level language modeling](https://github.com/pytorch/examples/tree/master/word_language_model) example from the excellent [PyTorch Examples](https://github.com/pytorch/examples) repo. Example usage: 

```
python train_dynamics.py --cuda --embedding-model 'in' --data 'a' --batch-size 256 --dropout 0.1
```
Please see the [`scripts`](https://github.com/eminorhan/temporal-prediction/tree/master/scripts) folder for other usage examples.

