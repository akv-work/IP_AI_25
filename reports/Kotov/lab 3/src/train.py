import os
import yaml
import torch
from ultralytics import YOLO

DATASET_DIR = os.path.abspath('dataset')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

EPOCHS = 3
IMGSZ = 320
BATCH = 8
FRACTION = 0.3

device = 0 if torch.cuda.is_available() else 'cpu'

with open(YAML_PATH, 'r', encoding='utf-8') as f:
    data_cfg = yaml.safe_load(f)

data_cfg['path'] = DATASET_DIR
if os.path.isdir(os.path.join(DATASET_DIR, 'train', 'images')):
    data_cfg['train'] = 'train/images'
if os.path.isdir(os.path.join(DATASET_DIR, 'valid', 'images')):
    data_cfg['val'] = 'valid/images'
elif os.path.isdir(os.path.join(DATASET_DIR, 'val', 'images')):
    data_cfg['val'] = 'val/images'

has_test = os.path.isdir(os.path.join(DATASET_DIR, 'test', 'images'))
if has_test:
    data_cfg['test'] = 'test/images'

with open(YAML_PATH, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data_cfg, f, sort_keys=False, allow_unicode=True)

model = YOLO('yolo12s.pt')

model.train(
    data=YAML_PATH,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    fraction=FRACTION,
    device=device,
    plots=True
)

eval_split = 'test' if has_test else 'val'
metrics = model.val(data=YAML_PATH, split=eval_split, imgsz=IMGSZ, device=device)

print(f'\nОценка на выборке ({eval_split}):')
print(f'mAP@50:    {metrics.box.map50:.4f}')
print(f'mAP@50-95: {metrics.box.map:.4f}')