# Looper config

The looper provide a global self-play framework for the GuessWhat?! game including the three models (Oracle/Qgen/Guesser)
Therefore, it enable to use RL methods to fine-tune the models

The configuration file is divided into three parts:
 - QGen model definition
 - QGen training
 - Others

The keyword "loop" refers to the QGen game constraints:
```
"loop": {
    "max_question": 5,  # maximum number of questions per game
    "max_depth" : 12,   # maximum number of words per question
    "beam_k_best" : 20  # number of beam used for beam-search
  },
```

The "optimizer" key refers to the training hyperparameters for the RL:

```
  "optimizer": {
    "no_epoch": int,            # the number of traiing epoches
    "learning_rate": float,     # SGD initial learning rate
    "batch_size": int,          # training batch size
    "clip_val": int             # gradient clip to avoid RNN gradient explosion
  },
 ```

Other parameters can be set such as:

```
  "seed": -1                                       # define the training seed; -1 -> random seed
 ```