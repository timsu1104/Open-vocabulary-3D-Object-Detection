"""
Generate categories file for RegionCLIP. 
"""
from argparse import ArgumentParser

# {'synonyms': ['cabinet'], 'def': 'a dispenser that holds a substance under pressure', 'id': 1, 'synset': 'cabinet', 'name': 'cabinet', 'frequency': 'c'}

def gen(cls_file, out_file):
    with open(cls_file, 'r') as f:
        classes = f.read().splitlines()

    c = []
    for i, _c in enumerate(classes):
        d = {'synonyms': [_c], 'def': '', 'id': i+1, 'synset': _c, 'name': _c, 'frequency': 'c'}
        c.append(d)

    LVIS_CATEGORIES = repr(c) + "  # noqa"
    with open(out_file, "wt") as f:
        f.write(f"LVIS_CATEGORIES = {LVIS_CATEGORIES}")
    
if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--dataset", default="sunrgbd", type=str)
    args = parser.parse_args()
    
    OUTPUT_PATH = '/home/zhengyuan/packages/RegionCLIP/detectron2/data/datasets/{}_categories.py'.format(args.dataset)
    CLASSES_FILE = '/home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_{}.txt'.format(args.dataset)
    
    gen(CLASSES_FILE, OUTPUT_PATH)