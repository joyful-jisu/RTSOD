from PIL import Image, ImageDraw
import torch
from torch import nn
from src.dfine.dfine import DFINE
from src.dfine.dfine_decoder import DFINETransformer
from src.dfine.hybrid_encoder import HybridEncoder
from src.dfine.postprocessor import DFINEPostProcessor
from src.nn.backbone.hgnetv2 import HGNetv2
from torchvision import transforms as T

model = DFINE(
            backbone=HGNetv2(name="B5", 
                            return_idx=[1, 2, 3], 
                            freeze_stem_only=True, freeze_at=0, freeze_norm=True, pretrained=True), 
            decoder=DFINETransformer(num_classes=12, feat_channels=[384, 384, 384], reg_scale=8), 
            encoder=HybridEncoder(hidden_dim=384, dim_feedforward=2048))

model.load_state_dict(torch.load("./last.pth")['model_state_dict'])
postprocessor = DFINEPostProcessor(
                num_classes=12,
                num_top_queries=300,
                remap_mscoco_category=False,
                use_focal_loss=True
            )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model.deploy()
        self.postprocessor = postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs
    
inference_model = Model().to('cuda')
img = Image.open("./samples/test1.png").convert('RGB')
w, h = img.size
orig_size = torch.tensor([[w, h]]).to('cuda')

transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
im_data = transforms(img).unsqueeze(0).to('cuda')

output = inference_model(im_data, orig_size)
labels, boxes, scores = output

def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i]
        box = boxes[i]
        scrs = scr

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill='blue',
            )

        im.save('torch_results.jpg')

draw([img], labels, boxes, scores)