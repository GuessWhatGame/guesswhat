# Guesser config

The guesser task requires to select an object within a list of objects object given a dialogue and a picture.

The configuration file is divided into three parts:
 - Guesser model definition
 - Guesser training
 - Others

In this model, objects are encoded by there spatial information and their category.

The keyword "model" refers to the model architecture of the guesser:
```
"model": {

    "word_emb_dim": 512,   # dimension of the word embedding for the dialogue
    "num_lstm_units": 512, # dimension of the LSTM for the dialogue

    "cat_emb_dim": 256,    # dimension of the object category embedding
    "no_categories": 90    # number of object category (90 for MS coco)
    "spat_dim": 8,         # dimension of the spatial information
    "obj_mlp_units": 512,  # number of hidden units to build the full object embedding

    "dialog_emb_dim": 512, # Projection size for the dialogue and the objects
  },
```

The keyword "optimizer" key refers to the training hyperparameters:


```
  "optimizer": {
    "no_epoch": int,            # the number of traiing epoches
    "learning_rate": float,     # Adam initial learning rate
    "batch_size": int,          # training batch size
    "clip_val": int             # gradient clip to avoid RNN gradient explosion
  },
 ```

Other parameters can be set such as:

```
  "seed": -1                                       # define the training seed; -1 -> random seed
 ```