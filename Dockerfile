FROM tensorflow/tensorflow:latest-py3

WORKDIR /usr/src/guesswhat

COPY . /usr/src/guesswhat

RUN apt-get update && apt-get install -y wget \
    && pip install nltk tensorflow tqdm image \
    && mkdir data data/img data/img/raw data/img/ft_vgg_img data/img/ft_vgg_crop \
             out out/oracle out/guesser out/qgen out/looper \
    && wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz -P data \
    && wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz -P data \
    && wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz -P data \
    && wget www.florian-strub.com/github/ft_vgg_img.zip -P data/img \
    && unzip data/img/ft_vgg_img.zip -d data/img \
    && wget http://florian-strub.com/github/pretrained_models.tf1-3.zip \
    && unzip pretrained_models.tf1-3.zip \
    && cp out/dict.json data/dict.json

ENV PYTHONPATH "/usr/src/guesswhat/src:$PYTHONPATH"
