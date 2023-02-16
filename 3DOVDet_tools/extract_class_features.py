"""
Generate text class embeddings. 

Example Usage: 
python extract_class_features.py --tag sunrgbd
"""

import torch
import clip
from detectron2.data.datasets.clip_prompt_utils import pre_tokenize
from tqdm import tqdm
from argparse import ArgumentParser

def extract(tag: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load("RN50x4", device=device)

    categories = open('/home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_{}.txt'.format(tag), 'r').readlines()
    categories = [x[:-1] if x[-1] == '\n' else x for x in categories]

    texts = pre_tokenize(categories).to(device)

    concept_feats = []
    with torch.no_grad():
        for text in tqdm(texts):
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(0)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
            concept_feats.append(text_features.cpu())
    concept_feats = torch.stack(concept_feats, 0)

    print(f"{concept_feats.size()}")

    torch.save(concept_feats,'/home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_{}.pth'.format(tag))

if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--tag", default="sunrgbd", type=str)
    args = parser.parse_args()
    
    extract(args.tag)