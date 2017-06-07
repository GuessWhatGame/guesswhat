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

    def __init__(self, id, object_id, picture, objects, qas, status):
        self.dialogue_id = id
        self.object_id = object_id
        self.picture = Picture(picture["id"], picture["width"], picture["height"], picture["coco_url"])
        self.objects = []
        for o in objects:
            new_obj = Object(o['id'],
                             o['category'],
                             o['category_id'],
                             Bbox(o['bbox'], picture["width"], picture["height"]),
                             o['area'])

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status


class Picture:
    def __init__(self, id, width, height, url):
        self.id = id
        self.width = width
        self.height = height
        self.url = url
        self.path = None
        self.fc8 = None

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
    def __init__(self, id, category, category_id, bbox, area):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.fc8 = None



class Dataset(AbstractDataset):
    """Loads the dataset."""
    def __init__(self, folder, which_set, fc8_img=None, fc8_crop=None, image_folder=None):
        file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        if image_folder is not None:
            self.image_folder = image_folder
        else:
            self.image_folder = os.path.join(folder, 'guesswhat_images')

        with gzip.open(file) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game(id=game['id'],
                         object_id=game['object_id'],
                         objects=game['objects'],
                         qas=game['qas'],
                         picture=game['image'],
                         status=game['status'])

                if fc8_img:
                    g.picture.fc8 = fc8_img[g.picture.id]

                if fc8_crop:
                    g.object.fc8 = fc8_crop[g.picture.id]

                # TODO: check if path exist?
                g.picture.path = os.path.join(self.image_folder, str(g.picture.id) + '.jpg')
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
        self.image_folder = dataset.image_folder
        super(OracleDataset, self).__init__(new_games)

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
