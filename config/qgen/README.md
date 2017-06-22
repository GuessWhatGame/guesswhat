# QGen config

The qgen task requires to produce a new question given a picture and the history of dialogues.

The configuration file is divided into three parts:
 - QGen model definition
 - QGen training
 - Others

The keyword "model" refers model configuration of the qgen:
```
"model": {

    "word_embedding_size": int,         # dimension of the word embedding for the dialogue
    "num_lstm_units": int,              # dimension of the LSTM for the dialogue
    "picture_embedding_size": int,      # dimension of the picture projection

    "image": {                          # configuration of the inout image
      "image_input": "features"/"raw",  # select the image inputs: raw vs feature
      "dim": list(int)                  # provide the image input dimension
    }

  },
```

The "optimizer" key refers to the training hyperparameters:


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