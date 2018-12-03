import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier, BatchifierSplitMode, batchifier_split_helper

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image, scaled_crop_and_pad
from generic.data_provider.nlp_utils import padder, padder_3d
from itertools import chain

import copy


class GuesserCropBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, ignore_NA=False, status=list()):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.ignore_NA = ignore_NA
        self.glove = glove

    def split(self, games):

        new_games = []

        for game in games:
            for i, obj in enumerate(game.objects):
                new_game = copy.copy(game)
                new_game.object = obj
                new_game.object_id = obj.id
                new_game.is_full_dialogue = (obj.id == game.object.id)  # hack!!! is_full_dialogue == is_object_to_find

                new_games.append(new_game)

        return new_games

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        if self.ignore_NA:
            games = [g for g in games if g.answers[-1] != "N/A"]

        return games

    def apply(self, games):
        sources = self.sources
        tokenizer = self.tokenizer
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            image = game.image

            if 'question' in sources:
                questions = []
                for q, a in zip(game.questions, game.answers):
                    questions.append(tokenizer.encode(q))
                    questions.append(tokenizer.encode(a, is_answer=True))
                questions.append([self.tokenizer.stop_dialogue])
                batch['question'].append(list(chain.from_iterable(questions)))

            if 'glove' in self.sources:
                questions = []
                for q, a in zip(game.questions, game.answers):
                    questions.append(self.tokenizer.tokenize_question(q))
                    questions.append([self.tokenizer.format_answer(a)])
                questions = list(chain.from_iterable(questions))
                questions += [self.tokenizer.stop_dialogue]
                glove_vectors = self.glove.get_embeddings(questions)
                batch['glove'].append(glove_vectors)

            if 'answer' in sources:
                answer = [0, 0]
                answer[int(game.is_full_dialogue)] = 1 # ugly tempo hack -> if is_full_dialogue == true then
                batch['answer'].append(answer)

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())

            if 'image' in sources:
                batch['image'].append(image.get_image())

            if 'image_mask' in sources:
                assert "image" in batch, "mask input require the image source"
                mask = game.object.get_mask()

                ft_width, ft_height = batch['image'][-1].shape[1], \
                                      batch['image'][-1].shape[0]  # Use the image feature size (not the original img size)

                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['image_mask'].append(np.array(mask))

            if 'crop_mask' in sources:
                assert "crop" in batch, "mask input require the crop source"
                cmask = game.object.get_mask()

                ft_width, ft_height = batch['crop'][-1].shape[1], \
                                      batch['crop'][-1].shape[0]  # Use the crop feature size (not the original img size)

                cmask = scaled_crop_and_pad(raw_img=Image.fromarray(cmask), bbox=game.object.bbox, scale=game.object.crop_scale)
                cmask = resize_image(cmask, height=ft_height, width=ft_width)
                batch['crop_mask'].append(np.array(cmask))

        # Pad the questions
        if 'question' in sources:
            batch['question'], batch['seq_length'] = padder(batch['question'], padding_symbol=tokenizer.word2i['<padding>'])

        if 'glove' in sources:
            # (?, 16, 300)   (batch, max num word, glove emb size)
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch
