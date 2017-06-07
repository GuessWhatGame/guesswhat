import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import (resize_image, get_spatial_feat,
                                                         scaled_crop_and_pad)
from generic.data_provider.nlp_utils import padder

answer_dict = \
    {'Yes': np.array([1, 0, 0], dtype=np.int32),
       'No': np.array([0, 1, 0], dtype=np.int32),
       'N/A': np.array([0, 0, 1], dtype=np.int32)
    }

def preprocess_img(raw_img, width, heigth, kwargs):

    img = resize_image(raw_img,width, heigth)
    img = np.array(img, dtype=np.float32)

    if "channel_mean" in kwargs:
        img -= kwargs["channel_mean"][None, None, :]

    return img

class OracleBatchifier(AbstractBatchifier):

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
        sources = self.sources
        tokenizer = self.tokenizer
        kwargs = self.kwargs
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            picture = game.picture

            if 'question' in sources:
                assert  len(game.questions) == 1
                batch['question'].append(tokenizer.apply(game.questions[0]))

            if 'answer' in sources:
                assert len(game.answers) == 1
                batch['answer'].append(answer_dict[game.answers[0]])

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, picture.width, picture.height)
                batch['spatial'].append(spat_feat)

            if 'crop_fc8' in sources:
                batch['crop_fc8'].append(game.object.fc8)

            if 'picture_fc8' in sources:
                batch['picture_fc8'].append(picture.fc8)

            # Load picture if required
            if any([x in sources for x in ('picture_raw', 'crop')]):
                raw_img = Image.open(game.picture.path).convert('RGB')

                if 'picture_raw' in sources:
                    img = preprocess_img(raw_img, kwargs["image_w"], kwargs["image_h"], kwargs)
                    batch['picture_raw'].append(img)

                if 'crop' in sources:
                    crop = scaled_crop_and_pad(raw_img=raw_img, bbox=game.object.bbox, scale=kwargs["scale"])
                    crop = preprocess_img(crop, kwargs["crop_width"], kwargs["crop_height"], kwargs)
                    batch['crop'].append(crop)

        # pad the questions
        if 'question' in sources:
            batch['question'], batch['seq_length'] = padder(batch['question'], padding_symbol=tokenizer.word2i['<padding>'])

        return batch



