import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
import numpy as np
from PIL import Image
import os


class TvidDataset(data.Dataset):

    CLASSES = ['bird', 'car', 'dog', 'lizard', 'turtle']

    def __init__(self, root, mode):

        assert mode in ['train', 'test']
        if not os.path.isabs(root):
            root = os.path.expanduser(root) if root[0] == '~' else os.path.abspath(root)

        self.images = []
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) # 图片处理工具

        for c, cls in enumerate(self.CLASSES):
            dir = os.path.join(root, cls) # 图片所在文件目录
            ann = os.path.join(root, cls + '_gt.txt') # 加载标注文件

            with open(ann) as f:
                for i, line in enumerate(f):
                    if (mode == 'train' and i >= 150) or i >= 180:
                        break # 180之后的图片存在瑕疵不适合训练
                    if mode == 'test' and i < 150: # 后30个作为验证集
                        continue
                    idx, *xyxy = line.strip().split(' ') # 解析每一行数据
                    self.images.append({
                        'path': os.path.join(dir, '%06d.JPEG' % int(idx)),
                        'cls': c,
                        'bbox': [int(c) for c in xyxy]}) # 通过mode不同返回不同的数据集

    def __len__(self): # 自定义数据集需要重写下面两个函数
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = Image.open(img_info['path'])
        img, bbox = self.transforms(img), torch.tensor(img_info['bbox'])/128 #bbox归一化
        return img, {'cls': img_info['cls'], 'bbox': bbox}


class BoxHead(nn.Module):#pending
    def __init__(self, lengths, num_classes):
        super(BoxHead, self).__init__()
        self.cls_score = nn.Sequential(*tuple([
            module for i in range(len(lengths) - 1)
            for module in (nn.Linear(lengths[i], lengths[i + 1]), nn.ReLU())]
            + [nn.Linear(lengths[-1], num_classes)]))

        self.bbox_pred = nn.Sequential(*tuple([
            module for i in range(len(lengths) - 1)
            for module in (nn.Linear(lengths[i], lengths[i + 1]), nn.ReLU())]
            + [nn.Linear(lengths[-1], 4)]))

    def forward(self, x):
        logits = self.cls_score(x)
        bbox = self.bbox_pred(x)
        return logits, bbox


class Detector(nn.Module):
    def __init__(self, lengths, num_classes):
        super(Detector, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
	# for param in resnet.parameters():
 	#	 param.requires_grad = False # 这一步能固定resnet的参数，用来迁移学习，训练速度也会更快
        # 输入in_channel=3, out_features=1000，去掉后面两层后输出为2048*4*4长度的数据
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity() # 去除最后两层，这里把最后两层用单位层替代，起到同样的效果
        self.backbone = resnet
        self.box_head = BoxHead(lengths, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # B, 2048*4*4
        # x = x.flatten(1)
        logits, bbox = self.box_head(x)
        return logits, bbox

def compute_iou(bbox1, bbox2):
    if isinstance(bbox1, torch.Tensor):
        bbox1 = bbox1.detach().numpy()
    if isinstance(bbox2, torch.Tensor):
        bbox2 = bbox2.detach().numpy()

    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    ix1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    ix2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    iy1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    iy2 = np.minimum(bbox1[:, 3], bbox2[:, 3])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    union = area1 + area2 - inter
    return inter / union

def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    for X, y in bar:
        X, gt_cls, gt_bbox = X.to(device), y['cls'].to(device), y['bbox'].to(device) # 数据转化
        logits, bbox = model(X)
        #criterion是一个字典类型，然后分别计算总的loss
        loss = criterion['cls'](logits, gt_cls) + 10 * criterion['box'](bbox, gt_bbox)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 更新weight
        correct += sum((torch.argmax(logits, axis=1) == gt_cls).cpu().detach().numpy() &
                       (compute_iou(bbox.cpu(), gt_bbox.cpu()) > iou_thr))
        total += len(X)

        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f}'
                            f' acc={correct / total * 100:.2f}'
                            f' loss={loss.item():.2f}')
    scheduler.step() # 更新lr

# 测试代码，输入模型，testloader, device，输出结果
def test_epoch(model, dataloader, device):
    model.eval()#设置评估模式
    with torch.no_grad():
        correct, correct_cls, total = 0, 0, 0
        for X, y in dataloader:
            X, gt_cls, gt_bbox = X.to(device), y['cls'].to(device), y['bbox'].to(device)
            '''?'''
            logits, bbox = model(X)
            correct += sum((torch.argmax(logits, axis=1) == gt_cls).cpu().detach().numpy() &
                           (compute_iou(bbox.cpu(), gt_bbox.cpu()) > iou_thr))
            correct_cls += sum((torch.argmax(logits, axis=1) == gt_cls))
            total += len(X)
            '''?'''
        print(f' val acc: {correct / total * 100:.2f}')

lr = 5e-3
batch = 32
epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" # cpu训练比gpu训练慢很多，不要轻易尝试
iou_thr = 0.5
seed=310
torch.manual_seed(seed) # 用来指定随机种子，以免不能复现结果

trainloader = data.DataLoader(TvidDataset(root='./data/tiny_vid', mode='train'),
                              batch_size=batch, shuffle=True, num_workers=0)
testloader = data.DataLoader(TvidDataset(root='./data/tiny_vid', mode='test'),
                             batch_size=batch, shuffle=True, num_workers=0)
model = Detector(lengths=(2048*4*4,2048,512),#这里还是没有搞懂
                 num_classes=5).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                            weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95,
                                            last_epoch=-1)
# 用来调整lr，lr随着epoch增加而逐渐减小
criterion = {'cls': nn.CrossEntropyLoss(), 'box': nn.L1Loss()}

for epoch in range(epochs):
    train_epoch(model, trainloader, criterion, optimizer,
                scheduler, epoch, device)  # 传入训练所需要的模型，数据加载器，评价函数，优化器，训练组数，设备型号
    test_epoch(model, testloader, device) # 传入测试所需要的模型，数据加载器，设备型号