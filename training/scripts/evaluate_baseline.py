"""Evaluate a selected checkpoint on the held-out test split."""
import argparse
import importlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_baseline import load_split, evaluate


def run(checkpoint_path, split_path, output, feature_module="baseline"):
    module=importlib.import_module(feature_module)
    data, digest=load_split(split_path, feature_module)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['contract']!=module.CONTRACT or checkpoint['split_hash']!=digest or checkpoint['labels']!=data['labels']:
        raise ValueError('Checkpoint does not match this split/feature contract')
    if output.exists():
        raise ValueError('Report already exists; choose a new output path')
    torch.set_num_threads(2)
    model=module.Classifier(len(data['labels']),checkpoint['hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    loader=DataLoader(module.Sequences(data['splits']['test'],Path(data['landmarks'])),batch_size=32,num_workers=0)
    score=evaluate(model,loader,len(data['labels']))
    score.update(labels=data['labels'],epoch=checkpoint['epoch'],evaluation_scope=data['evaluation_scope'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
    return score


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--feature-module',default='baseline')
    args=p.parse_args()
    result=run(args.checkpoint,args.split,args.output,args.feature_module)
    print(f'Test accuracy: {result["accuracy"]:.4f}; macro F1: {result["macro_f1"]:.4f}')
