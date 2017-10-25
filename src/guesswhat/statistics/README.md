# Guesswhat?! statistics

Those python scripts enable to plot a bunch of statistics given a dataset.

Given a dataset guesswhat.${name}.jsonl.gz, use the following scripts


```
name=train
python plot_them_all.py
    -data_dir data \
    -out_dir out \
    -name ${name} \
    -ignore_incomplete True
```

NB1: Note that train_qgen_reinforce will dump a valid file that you can plot.
NB2: If you want to dump the plots for the full dataset just concatenate the train/val/test dataset into a single file.

