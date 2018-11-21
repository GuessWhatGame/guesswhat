import numpy as np
import collections

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, scale_bbox
from generic.data_provider.nlp_utils import padder, padder_3d

from itertools import chain


class QuestionerBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer, sources, glove=None, status=list()):
        self.tokenizer = tokenizer
        self.sources = sources
        self.glove = glove
        self.status = status

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
            q_tokens = [self.tokenizer.encode(q) for q in game.questions]
            a_tokens = [self.tokenizer.encode(a, is_answer=True) for a in game.answers]

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

            batch["dialogue"].append(tokens)
            all_answer_indices.append(answer_indices)

            if 'glove' in self.sources:
                questions = []
                for q, a in zip(game.questions, game.answers):
                    questions.append(self.tokenizer.tokenize_question(q))
                    questions.append([self.tokenizer.format_answer(a)])
                questions = list(chain.from_iterable(questions))
                glove_vectors = self.glove.get_embeddings(questions)

                # Add start token and end token to glove_embedding (otherwise cannot concatenate word embedding and glove)
                glove_vectors.insert(0, np.zeros_like(glove_vectors[0]))
                glove_vectors.append(np.zeros_like(glove_vectors[0]))

                batch['glove'].append(glove_vectors)

            # Object embedding
            obj_spats, obj_cats = [], []
            for index, obj in enumerate(game.objects):

                bbox = obj.bbox
                spatial = get_spatial_feat(bbox, game.image.width, game.image.height)
                category = obj.category_id

                #                    1 point                 width         height
                bbox_coord = [bbox.x_left, bbox.y_upper, bbox.x_width, bbox.y_height]

                if obj.id == game.object_id:
                    batch['target_category'].append(category)
                    batch['target_spatial'].append(spatial)
                    batch['target_index'].append(index)
                    batch['target_bbox'].append(bbox_coord)

                obj_spats.append(spatial)
                obj_cats.append(category)
            batch['obj_spats'].append(obj_spats)
            batch['obj_cats'].append(obj_cats)

            # image
            img = game.image.get_image()
            if img is not None:
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

        # Pad dialogue tokens tokens
        batch['dialogue'], batch['seq_length_dialogue'] = padder(batch['dialogue'], padding_symbol=self.tokenizer.padding_token)
        seq_length = batch['seq_length_dialogue']
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
        batch['obj_seq_length'] = obj_length

        if 'glove' in self.sources:
            # (?, 16, 300)   (batch, max num word, glove emb size)
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch
