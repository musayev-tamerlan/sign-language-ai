"""Train on train only; select checkpoint on validation only."""
import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from baseline import CONTRACT, Classifier, Sequences, metrics

ROOT=Path(__file__).resolve().parents[2]


def load_split(path, feature_module="baseline"):
    module = importlib.import_module(feature_module)
    raw=path.read_bytes()
    data=json.loads(raw)
    if data['contract'] != module.CONTRACT or not data.get('complete'):
        raise ValueError('Incompatible or incomplete split')
    labels=data['labels']
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Invalid labels')
    paths, hashes=set(),set()
    root=Path(data['landmarks']).resolve()
    for name in ('train','validation','test'):
        rows=data['splits'][name]
        if not rows or {r['label_id'] for r in rows} != set(range(len(labels))):
            raise ValueError('Each split must cover every class')
        part_hashes=set()
        for row in rows:
            resolved=(root/row['path']).resolve()
            if not resolved.is_relative_to(root) or row['path'] in paths or row['sha256'] in hashes:
                raise ValueError('Unsafe path or overlapping splits')
            paths.add(row['path'])
            part_hashes.add(row['sha256'])
        hashes.update(part_hashes)
    return data, hashlib.sha256(raw).hexdigest()


def evaluate(model, loader, classes):
    model.eval()
    truth,predictions=[],[]
    with torch.inference_mode():
        for x,y in loader:
            predictions.extend(model(x).argmax(1).tolist())
            truth.extend(y.tolist())
    return metrics(truth,predictions,classes)


def sample_weights(rows, classes):
    counts = np.bincount([r['label_id'] for r in rows], minlength=classes)
    if len(counts) != classes or (counts == 0).any():
        raise ValueError('Every class needs training samples')
    return torch.tensor([1.0 / counts[r['label_id']] for r in rows], dtype=torch.double)


def class_weights(rows, classes, power):
    """Return mean-one inverse-frequency CE weights using training data only."""
    if not 0 <= power <= 1:
        raise ValueError("class weight power must be between 0 and 1")
    counts = np.bincount([row['label_id'] for row in rows], minlength=classes)
    if len(counts) != classes or (counts == 0).any():
        raise ValueError('Every class needs training samples')
    weights = counts.astype(np.float64) ** (-power)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train(split_path, output, epochs=40, batch_size=32, hidden=96, patience=7, threads=2,
          balanced_sampling=False, feature_module="baseline", class_weight_power=0.0):
    if min(epochs,batch_size,hidden,patience,threads)<1 or not 0 <= class_weight_power <= 1:
        raise ValueError('Training options must be positive')
    module = importlib.import_module(feature_module)
    data, split_hash=load_split(split_path, feature_module)
    torch.set_num_threads(threads)
    seed=data['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    root=Path(data['landmarks'])
    if output.exists():
        raise ValueError('Output already exists; choose a new experiment directory')
    print('Validating train/validation files before training...', flush=True)
    # Validate the snapshot before creating any checkpoint.
    for name in ('train','validation'):
        dataset=module.Sequences(data['splits'][name],root)
        for index in range(len(dataset)):
            dataset[index]
    output.mkdir(parents=True,exist_ok=False)
    (output/'splits.json').write_bytes(split_path.read_bytes())
    config = dict(seed=seed, epochs=epochs, batch_size=batch_size, hidden=hidden,
                  patience=patience, threads=threads, balanced_sampling=balanced_sampling,
                  split_hash=split_hash, contract=module.CONTRACT,
                  feature_module=feature_module, class_weight_power=class_weight_power)
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    model=module.Classifier(len(data['labels']),hidden)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    loss_fn=torch.nn.CrossEntropyLoss(
        weight=class_weights(data['splits']['train'], len(data['labels']), class_weight_power)
        if class_weight_power else None
    )
    generator=torch.Generator().manual_seed(seed)
    sampler = None
    if balanced_sampling:
        rows = data['splits']['train']
        sampler = WeightedRandomSampler(sample_weights(rows,len(data['labels'])),
                                        num_samples=len(rows), replacement=True, generator=generator)
    train_loader=DataLoader(module.Sequences(data['splits']['train'],root),batch_size=batch_size,
                            shuffle=sampler is None,sampler=sampler,generator=generator,num_workers=0)
    val_loader=DataLoader(module.Sequences(data['splits']['validation'],root),batch_size=batch_size,num_workers=0)
    best,stale,history=-1.0,0,[]
    for epoch in range(1,epochs+1):
        model.train(); total_loss=0.0
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(x),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss+=loss.item()*len(y)
        score=evaluate(model,val_loader,len(data['labels']))
        row=dict(epoch=epoch,loss=total_loss/len(train_loader.dataset),validation=score)
        history.append(row)
        print(f'Epoch {epoch}: loss={row["loss"]:.4f} val_accuracy={score["accuracy"]:.4f} val_macro_f1={score["macro_f1"]:.4f}',flush=True)
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if score['macro_f1']>best:
            best,stale=score['macro_f1'],0
            checkpoint=dict(state_dict=model.state_dict(),hidden=hidden,labels=data['labels'],contract=module.CONTRACT,split_hash=split_hash,epoch=epoch,validation=score,torch_version=str(torch.__version__),config=config)
            torch.save(checkpoint,output/'best.tmp')
            (output/'best.tmp').replace(output/'best.pt')
        else:
            stale+=1
            if stale>=patience:
                break
    (output/'completed.json').write_text(json.dumps(dict(epochs_completed=len(history),
        best_validation_macro_f1=best, stopped_early=len(history)<epochs)),encoding='utf-8')
    return output/'best.pt'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,default=ROOT/'training/data/generated/splits.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--hidden',type=int,default=96)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--balanced-sampling',action='store_true')
    p.add_argument('--class-weight-power', type=float, default=0.0)
    p.add_argument('--feature-module',default='baseline')
    args=p.parse_args()
    print(train(args.split,args.output,args.epochs,args.batch_size,args.hidden,args.patience,args.threads,args.balanced_sampling,args.feature_module,args.class_weight_power))


if __name__=='__main__':
    main()
