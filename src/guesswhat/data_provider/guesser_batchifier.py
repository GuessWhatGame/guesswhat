import numpy as np
import collections
from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat
from generic.data_provider.nlp_utils import padder, padder_3d


class GuesserBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, status=list()):
        self.sources = sources
        self.status = status
        self.tokenizer = tokenizer
        self.glove = glove

    def filter(self, games):

        if len(self.status) > 0:
            games = [g for g in games if g.status in self.status]

        return games

    def apply(self, games, skip_targets=False):

        batch = collections.defaultdict(list)
        batch["raw"] = games
        batch_size = len(games)

        for i, game in enumerate(games):

            # Flattened question answers
            q_tokens = [self.tokenizer.encode(q, add_stop_token=True) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

            tokens = [self.tokenizer.start_token]  # Add start token
            for q_tok, a_tok in zip(q_tokens, a_tokens):
                tokens += q_tok
                tokens += a_tok
            if tokens[-1] != self.tokenizer.stop_dialogue:
                tokens += [self.tokenizer.stop_dialogue]  # Add STOP token

            batch["dialogue"].append(tokens)

            if 'glove' in self.sources:
                words = self.tokenizer.decode(batch['dialogue'][i])
                glove_vectors = self.glove.get_embeddings(words)
                batch['glove'].append(glove_vectors)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):

                bbox = obj.bbox
                spatial = get_spatial_feat(bbox, game.image.width, game.image.height)
                category = obj.category_id

                #                    1 point                 width         height
                bbox_coord = [bbox.x_left, bbox.y_upper, bbox.x_width, bbox.y_height]

                if obj.id == game.object.id and not skip_targets:
                    batch['target_category'].append(category)
                    batch['target_spatial'].append(spatial)
                    batch['target_index'].append(index)
                    batch['target_bbox'].append(bbox_coord)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spat'].append(obj_spats)
            batch['obj_cat'].append(obj_cats)

            # image
            if 'image' in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)

        # Pad objects
        batch['obj_spat'], obj_length = padder_3d(batch['obj_spat'])
        batch['obj_cat'], obj_length = padder(batch['obj_cat'])
        batch['obj_seq_length'] = obj_length

        if 'glove' in self.sources:
            # (?, 16, 300)   (batch, max num word, glove emb size)
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch
