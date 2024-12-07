import torch
import torchvision.transforms.v2 as T
import tqdm
from src.data.dataloader import BatchImageCollateFunction
from src.data.coco_dataset import CocoDetection
from src.data.transforms._transforms import ConvertBoxes, ConvertPILImage
from src.data.transforms.container import Compose
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.model import Model
from src.nn.criterion import Criterion
from src.nn.decoder import Transformer
from src.nn.hybrid_encoder import HybridEncoder
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from src.nn.matcher import HungarianMatcher
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import BatchSampler, RandomSampler


train_data_source = CocoDetection(
        img_folder="./dataset/images/train",
        ann_file="./dataset/annotations/instances_train.json",
        return_masks=False,
        transforms=Compose(
            ops=[
                T.RandomPhotometricDistort(p=0.5),
                T.RandomZoomOut(fill=0),
                T.RandomIoUCrop(),
                T.SanitizeBoundingBoxes(min_size=1),
                T.RandomHorizontalFlip(),
                T.Resize(size=[640, 640], ),
                T.SanitizeBoundingBoxes(min_size=1),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True),
        ],
        policy={'name': 'stop_epoch', 
                'epoch': 72, 
                'ops': ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']}),
) 
val_data_source = CocoDetection(
    img_folder="./dataset/images/val",
    ann_file="./dataset/annotations/instances_val.json",
    return_masks=False,
    transforms=Compose(ops=[
                T.Resize(size=[640, 640]),
                ConvertPILImage(dtype='float32', scale=True),
            ]),
) 

train_dataloader = DataLoader(
    dataset=train_data_source,
    num_workers=4,
    prefetch_factor=2,
    pin_memory=False,
    pin_memory_device="",
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
    batch_sampler=BatchSampler(RandomSampler(train_data_source), batch_size=4, drop_last=True),
    generator=None,
    collate_fn=BatchImageCollateFunction(),
    persistent_workers=False,
)

model = Model(
            backbone=HGNetv2(name="B5", 
                            return_idx=[1, 2, 3], 
                            freeze_stem_only=True, 
                            freeze_at=0, 
                            freeze_norm=True, 
                            pretrained=True), 
            decoder=Transformer(num_classes=12, feat_channels=[384, 384, 384], reg_scale=8), 
            encoder=HybridEncoder(hidden_dim=384, dim_feedforward=2048))

model.train()
model.to('cuda')

criterion = Criterion(
    matcher=HungarianMatcher(weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                             alpha=0.25, gamma=2.0),
    losses=['vfl', 'boxes', 'local'],
    weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2, 'loss_fgl': 0.15, 'loss_ddf': 1.5},
    alpha=0.75)

criterion.to('cuda')

optimizer = AdamW(model.parameters(), lr=0.000125, weight_decay=1e-4)

for epoch in range(1):
    for i, (images, targets) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        images = images.to('cuda')
        targets = [{k: v.to("cuda") for k, v in t.items()} for t in targets]
        outputs = model(images, targets=targets)
        
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tqdm.tqdm.write(f"Epoch {epoch}, Step {i}, Loss: {loss.item()}")
    print(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, "./last.pth")
    print(f"Model saved at epoch {epoch}.")
