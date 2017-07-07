import gzip
import json
import os
import copy


from generic.data_provider.dataset import AbstractDataset

def lazy_property(fn):
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


class Game:

    def __init__(self, id, object_id, picture, objects, qas, status, image_loader, crop_loader):
        self.dialogue_id = id
        self.object_id = object_id
        self.picture = Picture(picture["id"],
                               picture["width"],
                               picture["height"],
                               picture["coco_url"],
                               image_loader=image_loader)
        self.objects = []
        for o in objects:

            new_obj = Object(o['id'],
                             o['category'],
                             o['category_id'],
                             Bbox(o['bbox'], picture["width"], picture["height"]),
                             o['area'],
                             crop_loader=crop_loader if o['id'] == object_id
                                        else None
                             )

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status


class Picture:
    def __init__(self, id, width, height, url, image_loader=None):
        self.id = id
        self.width = width
        self.height = height
        self.url = url
        self.path = None
        self.fc8 = None

        self.image_loader = image_loader
        if image_loader is not None:
            self.image_loader = image_loader.preload(id)

    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(self.id)
        else:
            return None





class Bbox:
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        self.x_width = bbox[2]
        self.y_height = bbox[3]

        self.x_left = bbox[0]
        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5*self.x_width
        self.y_center = self.y_lower + 0.5*self.y_height

        self.coco_bbox = bbox


class Object:
    def __init__(self, id, category, category_id, bbox, area, crop_loader):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area

        if crop_loader is not None:
            self.crop_loader = crop_loader.preload(id)

    def get_crop(self, **kwargs):
        return self.crop_loader.get_image(self.id, **kwargs)




class Dataset(AbstractDataset):
    """Loads the dataset."""
    def __init__(self, folder, which_set, image_loader=None, crop_loader=None):
        file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        with gzip.open(file) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game(id=game['id'],
                         object_id=game['object_id'],
                         objects=game['objects'],
                         qas=game['qas'],
                         picture=game['image'],
                         status=game['status'],
                         image_loader=image_loader,
                         crop_loader=crop_loader)

                games.append(g)

                # if len(games) > 200: break

        super(Dataset, self).__init__(games)


class OracleDataset(AbstractDataset):
    """
    Each game contains a single question answer pair
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        for g in old_games:
            new_games += self.split(g)
        super(OracleDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, image_loader=None, crop_loader=None):
        return cls(Dataset(folder, which_set, image_loader, crop_loader))

    def split(self, game):
        games = []
        for i, q, a in zip(game.question_ids, game.questions, game.answers):
            new_game = copy.copy(game)
            new_game.questions = [q]
            new_game.question_ids = [i]
            new_game.answers = [a]
            games.append(new_game)
        return games


def dump_samples_into_dataset(data, save_path, tokenizer, name="model"):

    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for i, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]

            sample = {}

            qas = []
            start = 1
            for i, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                                word == tokenizer.no_token or \
                                word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:i - 1])
                    a = tokenizer.decode([dialogue[i]])
                    qas.append({"question": q, "answer": a[1:-1], "id":0})
                    start = i + 1

            sample["id"] = i
            sample["qas"] = qas
            sample["image"] = {
                "id": game.picture.id,
                "width": game.picture.width,
                "height": game.picture.height,
                "coco_url": game.picture.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')
