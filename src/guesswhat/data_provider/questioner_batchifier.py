import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import (resize_image, get_spatial_feat,
                                                         scaled_crop_and_pad)
from generic.data_provider.nlp_utils import padder, padder_3d

answer_dict = \
    {'Yes': np.array([1, 0, 0], dtype=np.int32),
    'No': np.array([0, 1, 0], dtype=np.int32),
    'N/A': np.array([0, 0, 1], dtype=np.int32)
    }


class QuestionerBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, status=list(), **kwargs):
        self.tokenizer = tokenizer
        self.sources = sources
        self.status = status
        self.kwargs = kwargs

    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games


    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        all_answer_indices = []
        for i, game in enumerate(games):

            batch['raw'].append(game)

            # Flattened question answers
            q_tokens = [self.tokenizer.apply(q) for q in game.questions]
            a_tokens = [self.tokenizer.apply(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add start token
            answer_indices = []
            cur_index = 0
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok

                # Compute index of answer in the full dialogue
                answer_indices += [cur_index + len(q_tok) + 1]
                cur_index = answer_indices[-1]

            tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogues"].append(tokens)
            all_answer_indices.append(answer_indices)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):
                spatial = get_spatial_feat(obj.bbox, game.picture.width, game.picture.height)
                category = obj.category_id

                if obj.id == game.object_id:
                    batch['targets_category'].append(category)
                    batch['targets_spatial'].append(spatial)
                    batch['targets_index'].append(index)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spats'].append(obj_spats)
            batch['obj_cats'].append(obj_cats)

            # image
            img = game.picture.get_image()
            if img is not None:
                if "images" not in batch:  # initialize an empty array for better memory consumption
                    batch["images"] = np.zeros((batch_size,) + img.shape)
                batch["images"][i] = img


        # Pad dialogue tokens tokens
        batch['dialogues'], batch['seq_length'] = padder(batch['dialogues'], padding_symbol=self.tokenizer.padding_token)
        seq_length = batch['seq_length']
        max_length = max(seq_length)

        # Compute the token mask
        batch['padding_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['padding_mask'][i, (seq_length[i] + 1):] = 0.

        # Compute the answer mask
        batch['answer_mask'] = np.ones((batch_size, max_length), dtype=np.float32)
        for i in range(batch_size):
            batch['answer_mask'][i, all_answer_indices[i]] = 0.

        # Pad objects
        batch['obj_spats'], obj_length = padder_3d(batch['obj_spats'])
        batch['obj_cats'], obj_length = padder(batch['obj_cats'])

        # Compute the object mask
        max_objects = max(obj_length)
        batch['obj_mask'] = np.zeros((batch_size, max_objects), dtype=np.float32)
        for i in range(batch_size):
            batch['obj_mask'][i, :obj_length[i]] = 1.0

        return batch




