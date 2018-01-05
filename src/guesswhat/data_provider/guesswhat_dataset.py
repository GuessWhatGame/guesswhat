import gzip
import json
import copy
import os

from PIL import ImageFont, ImageDraw
from PIL import Image as PImage


from generic.data_provider.dataset import AbstractDataset

# TODO find a cleaner way!
try:
    import cocoapi.PythonAPI.pycocotools.mask as cocoapi
    use_coco = True
except ImportError:
    print("Coco API was not detected - advanced segmentation features cannot be used")
    use_coco = False
    pass


class Game:

    def __init__(self, id, object_id, image, objects, qas, status, which_set, image_builder, crop_builder):
        self.dialogue_id = id
        self.object_id = object_id
        self.image = Image(id=image["id"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           which_set=which_set,
                           image_builder=image_builder)
        self.objects = []
        for o in objects:

            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

            self.objects.append(new_obj)
            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find

        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status

    def show(self, img_raw_dir, display_index=False, display_mask=False):
            image_path = os.path.join(img_raw_dir, self.image.filename)

            img = PImage.open(image_path)
            draw = ImageDraw.Draw(img)

            for i, obj in enumerate(self.objects):
                if display_index:
                    draw.text((obj.bbox.x_center, self.image.height-obj.bbox.y_center), str(i))
                if display_mask:
                    print("Show mask: Not yet implemented... sry")

            img.show()






class Image:
    def __init__(self, id, width, height, url, which_set, image_builder=None):
        self.id = id
        self.width = width
        self.height = height
        self.url = url

        self.image_loader = None
        if image_builder is not None:
            self.filename = "{}.jpg".format(id)
            self.image_loader = image_builder.build(id, which_set=which_set, filename=self.filename, optional=False)

    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(**kwargs)
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

    def __str__(self):
        return "center : {0:5.2f}/{1:5.2f} - size: {2:5.2f}/{3:5.2f}"\
            .format(self.x_center, self.y_center, self.x_width, self.y_height)


class Object:
    def __init__(self, id, category, category_id, bbox, area, segment, crop_builder, image, which_set):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment
 
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        # self.rle_mask = None
        # if use_coco:
        #     self.rle_mask = cocoapi.frPyObjects(self.segment,
        #                                         h=image.height,
        #                                         w=image.width)

        if crop_builder is not None:
            filename = "{}.jpg".format(image.id)
            self.crop_loader = crop_builder.build(id, filename=filename, which_set=which_set, bbox=bbox)

    # def get_mask(self):
    #     assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
    #     return cocoapi.decode(self.rle_mask)

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)




class Dataset(AbstractDataset):
    """Loads the dataset."""
    def __init__(self, folder, which_set, image_builder=None, crop_builder=None):
        file = '{}/guesswhat.{}.jsonl.gz'.format(folder, which_set)
        games = []

        self.set = which_set

        with gzip.open(file) as f:
            for line in f:
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))

                g = Game(id=game['id'],
                         object_id=game['object_id'],
                         objects=game['objects'],
                         qas=game['qas'],
                         image=game['image'],
                         status=game['status'],
                         which_set=which_set,
                         image_builder=image_builder,
                         crop_builder=crop_builder)

                games.append(g)

                #if len(games) > 200: break

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
    def load(cls, folder, which_set, image_builder=None, crop_builder=None):
        return cls(Dataset(folder, which_set, image_builder, crop_builder))

    def split(self, game):
        games = []
        for i, q, a in zip(game.question_ids, game.questions, game.answers):
            new_game = copy.copy(game)
            new_game.questions = [q]
            new_game.question_ids = [i]
            new_game.answers = [a]
            games.append(new_game)
        return games


class CropDataset(AbstractDataset):
    """
    Each game contains no question/answers but a new object
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        for g in old_games:
            new_games += self.split(g)
        super(CropDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, image_builder=None, crop_builder=None):
        return cls(Dataset(folder, which_set, image_builder, crop_builder))

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)
            new_game.questions = []
            new_game.question_ids = []
            new_game.answers = []
            new_game.object_id = obj.id

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games


def dump_samples_into_dataset(data, save_path, tokenizer, name="model"):

    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for id_game, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]
            prob_objects = d["prob_objects"]
            guess_object_id = d["guess_object_id"]

            sample = {}

            qas = []
            start  = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                                word == tokenizer.no_token or \
                                word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas),:len(game.objects)])
                    prob_obj = [str(round(p,3)) for p in prob_obj] # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id":0,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = id_game
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment" : [], #no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')
