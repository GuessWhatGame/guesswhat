import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image, scaled_crop_and_pad
from generic.data_provider.nlp_utils import padder, padder_3d
from itertools import chain


class OracleBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, ignore_NA=False, status=list(), split_mode=BatchifierSplitMode.NoSplit):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.ignore_NA = ignore_NA
        self.glove = glove
        self.split_mode = split_mode

    def split(self, games):
        return batchifier_split_helper(games, split_mode=self.split_mode)

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        if self.ignore_NA:
            games = [g for g in games if g.answers[-1] != "N/A"]

        return games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games
        batch_size = len(games)

        for i, game in enumerate(games):

            if 'question' in self.sources:
                questions = []
                for q, a in zip(game.questions[:-1], game.answers[:-1]):
                    questions.append(self.tokenizer.encode(q, add_stop_token=True))
                    questions.append(self.tokenizer.encode(a, is_answer=True))
                questions.append(self.tokenizer.encode(game.questions[-1], add_stop_token=True))
                batch['question'].append(list(chain.from_iterable(questions)))

            if 'glove' in self.sources:
                words = self.tokenizer.decode(batch['question'][i])
                glove_vectors = self.glove.get_embeddings(words)
                batch['glove'].append(glove_vectors)

            if 'answer' in self.sources and not skip_targets:
                batch['answer'].append(self.tokenizer.encode_oracle_answer(game.answers[-1], sparse=False))

            if 'category' in self.sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in self.sources:
                spat_feat = get_spatial_feat(game.object.bbox, game.image.width, game.image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in self.sources:
                crop = game.object.get_crop()
                if "crop" not in batch:  # initialize an empty array for better memory consumption
                    batch["crop"] = np.zeros((batch_size,) + crop.shape)
                batch["crop"][i] = crop

            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            if 'image_mask' in self.sources:
                assert "image" in batch, "mask input require the image source"
                mask = game.object.get_mask()

                ft_width, ft_height = batch['image'][-1].shape[1], \
                                      batch['image'][-1].shape[0]  # Use the image feature size (not the original img size)

                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['image_mask'].append(np.array(mask))

            if 'crop_mask' in self.sources:
                assert "crop" in batch, "mask input require the crop source"
                cmask = game.object.get_mask()

                ft_width, ft_height = batch['crop'][-1].shape[1], \
                                      batch['crop'][-1].shape[0]  # Use the crop feature size (not the original img size)

                cmask = scaled_crop_and_pad(raw_img=Image.fromarray(cmask), bbox=game.object.bbox, scale=game.object.crop_scale)
                cmask = resize_image(cmask, height=ft_height, width=ft_width)
                batch['crop_mask'].append(np.array(cmask))

        # Pad the questions
        if 'question' in self.sources:
            batch['question'], batch['seq_length'] = padder(batch['question'], padding_symbol=self.tokenizer.padding_token)

        if 'glove' in self.sources:
            # (?, 16, 300)   (batch, max num word, glove emb size)
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch
