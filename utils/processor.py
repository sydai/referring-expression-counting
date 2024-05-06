import json
import os
import random

DATA_ROOT = '/mnt/workstation/images' # update this to the path that contains 'rec-8k' folder

class DataProcessor(): 
    def __init__(self):
        self.image_path = os.path.join(DATA_ROOT, 'rec-8k')

        self.anno_file = 'anno/annotations.json'
        self.annotations = self.read_annotations()
    
        self.split_file = 'anno/splits.json'
        self.splits = self.read_splits()
        
        print(f"annotation file: {self.anno_file}\nsplit file: {self.split_file}")
        print(f"train: {len(self.splits['train'])}\nval: {len(self.splits['val'])}\ntest: {len(self.splits['test'])}")
        
        
    def read_annotations(self):
        annotations = {}
        with open(self.anno_file) as f:
            anno = json.load(f)
            for image_id, captions in anno.items():
                for caption, items in captions.items():
                    annotations[(image_id, caption)] = items

        return annotations

    def read_splits(self):
        with open(self.split_file) as f:
            splits = json.load(f)
            splits = {key: [tuple(x) for x in item] for key, item in splits.items()}
        
        return splits

    def get_image_path(self): 
        return self.image_path


    def get_anno_for_tuple(self, image_id, caption):
        return self.annotations[(image_id, caption)]
        
    def get_class_name(self, image_id, caption):
        class_name = self.annotations[(image_id, caption)]['class']
        return class_name
    
    def get_attr_name(self, image_id, caption):
        attr_name = self.annotations[(image_id, caption)]['attribute']
        return attr_name

    def get_type_name(self, image_id, caption):
        type_name = self.annotations[(image_id, caption)]['type']
        return type_name
    
    def get_split_type(self, image_id, caption):
        for split_type in self.splits.keys():
            if (image_id, caption) in self.splits[split_type]:
                return split_type

    def get_prompt_for_image(self, image_id_caption: tuple):
        assert type(image_id_caption) == tuple and len(image_id_caption) == 2, 'input must be a tuple of (image_id, caption)'

        _, caption = image_id_caption
        
        return [caption]
    
    def get_img_ids_for_split(self, split):
        return self.splits[split] # list of tuples (image_id, caption)

