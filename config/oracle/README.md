# Oracle config

The oracle task requires to produce a yes-no answer for any object within a picture given a natural language question.
Given an question, a picture and a specific object in the picture, the oracle must provide a valid yes/no answer

The configuration file is divided into four parts:
 - Oracle inputs
 - Oracle model definition
 - Oracle training
 - Others

There are several way way to encode the target object for the oracle.
Should one use the crop of the object, its spatial features, its category etc.
The keyword "inputs" enables to add/remove oracle inputs.

```
  "inputs": {
    "question": bool,  # Provide the question to the oracle
    "image": bool      # Provide the question to the oracle

    "category": bool,  # Provide the object category to the oracle
    "spatial": bool,   # Provide the spatial information of the object to the oracle
    "crop": bool,      # Provide the image crop of the object to the oracle
  },
```


The keyword  "model" refers to the oracle input configuration:
```
"model": {
    "question": {
      "embedding_dim": int,             # the dimension of the word embedding
      "no_LSTM_hiddens": int            # the number f hidden units in the LSTM encoding the question
    },

    "category": {
      "n_categories": int,              # the number of object categories that are available (MS Coco has 90 categories)
      "embedding_dim": int              # the dimension of the category embedding
    },

    "image": {
      "image_input": "raw"/"features",  # the type of image inputs (*raw*  images VS preprocess image *features*)
      "dim": list                       # the dimensin of image inputs. Ex fc7: [4000], resnet-224*224: [7,7,2048] etc.
    },

    "crop": {
      "image_input": "raw"/"features",  # the type of crop inputs (*raw* images VS preprocess image *features*)
      "dim": list(int)                  # the dimensin of crop inputs. Ex fc7: [4000], resnet-224*224: [7,7,2048] etc.
    }

    "MLP": {
      "num_hiddens": int                # the number of hidden units of the multilayer-perception before the softmax
    },

  },
```

The keyword  "optimizer" refers to the training hyperparameters:


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
  "status": ["success", "failure", "incomplete"],  # The oracle can be trained on either the full GW dataset or a subset
  "seed": -1                                       # define the training seed; -1 -> random seed
 ```