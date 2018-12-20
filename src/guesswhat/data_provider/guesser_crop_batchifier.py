import numpy as np
from PIL import Image
import collections

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image, scaled_crop_and_pad
from generic.data_provider.nlp_utils import padder, padder_3d
from itertools import chain

import copy


class GuesserCropBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, ignore_NA=False, status=list()):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.ignore_NA = ignore_NA
        self.glove = glove

    def split(self, games):

        new_games = []

        for game in games:
            for i, obj in enumerate(game.objects):
                new_game = copy.deepcopy(game)
                new_game.object = obj
                new_game.user_data["is_target_object"] = (obj.id == game.object.id)

                new_games.append(new_game)

        return new_games

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

            image = game.image

            if 'question' in self.sources:
                questions = []
                for q, a in zip(game.questions, game.answers):
                    questions.append(self.tokenizer.encode(q, add_stop_token=True))
                    questions.append(self.tokenizer.encode(a, is_answer=True))
                if questions[-1] != self.tokenizer.stop_dialogue:
                    questions.append([self.tokenizer.stop_dialogue])
                batch['question'].append(list(chain.from_iterable(questions)))

            if 'glove' in self.sources:
                words = self.tokenizer.decode(batch['question'][i])
                glove_vectors = self.glove.get_embeddings(words)
                batch['glove'].append(glove_vectors)

            if 'answer' in self.sources and not skip_targets:
                answer = [0, 0]
                answer[int(game.user_data["is_target_object"])] = 1  # False: 0 / True: 1
                batch['answer'].append(answer)

            if 'category' in self.sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in self.sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
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
