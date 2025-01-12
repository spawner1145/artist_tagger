import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score
import warnings
import cv2

base_dir = 'D:/tagger/artist_tagger/train_example_folder'  # tags.csv所在的文件夹
# 预测目录中的图像
img_dir = 'D:/tagger/artist_tagger/input_images'  # 替换为要预测的图片所在的目录
# 模型路径
model_path = 'D:/tagger/artist_tagger/tagger.pth'
# 设置阈值，不支持使用自动阈值(因为默认你没有验证集), 输入0到1的阈值
threshold = 0.3

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

def convert_images_to_jpg(input_dir):
    supported_formats = ['.bmp', '.tiff', '.webp', '.gif']
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.lower().endswith('.jpg'):
            continue
        try:
            if filename.lower().endswith('.gif'):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    output_path = os.path.splitext(file_path)[0] + '.jpg'
                    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                os.remove(file_path)
            elif filename.lower().endswith(tuple(supported_formats)):
                img = Image.open(file_path).convert("RGB")
                output_path = os.path.splitext(file_path)[0] + '.jpg'
                img.save(output_path, "JPEG")
                img.close()
                if file_path != output_path:
                    os.remove(file_path)
            else:
                print(f"Unsupported file type: {filename}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

class ArtStyleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mlb=None):
        self.annotations = pd.read_csv(csv_file, encoding='latin1')
        self.img_dir = img_dir
        self.transform = transform
        self.mlb = mlb
        # 将字符串格式的列表转换回列表
        self.annotations['tags'] = self.annotations['tags'].apply(lambda x: x.split(','))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        try:
            image = Image.open(img_name).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image {img_name}: {e}")
            return None
        
        if self.transform:
            image = self.transform(image)

        labels = self.annotations.iloc[index]['tags']
        labels = self.mlb.transform([labels])[0]  # 直接传入标签列表
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # 过滤掉None值
    return torch.utils.data.dataloader.default_collate(batch)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MultiLayerAttentionResNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLayerAttentionResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        
        # 提取 ResNet 的各层
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 添加 SE Block 和空间注意力模块到每一层
        self.se_block1 = SEBlock(channel=self.layer1[-1].conv3.out_channels)
        self.spatial_attention1 = SpatialAttentionModule()
        
        self.se_block2 = SEBlock(channel=self.layer2[-1].conv3.out_channels)
        self.spatial_attention2 = SpatialAttentionModule()
        
        self.se_block3 = SEBlock(channel=self.layer3[-1].conv3.out_channels)
        self.spatial_attention3 = SpatialAttentionModule()
        
        self.se_block4 = SEBlock(channel=self.layer4[-1].conv3.out_channels)
        self.spatial_attention4 = SpatialAttentionModule()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加多层感知机（MLP）
        self.mlp = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.layer0(x)
        
        x = self.layer1(x)
        x = self.se_block1(x)
        x = x * self.spatial_attention1(x)
        
        x = self.layer2(x)
        x = self.se_block2(x)
        x = x * self.spatial_attention2(x)
        
        x = self.layer3(x)
        x = self.se_block3(x)
        x = x * self.spatial_attention3(x)
        
        x = self.layer4(x)
        x = self.se_block4(x)
        x = x * self.spatial_attention4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        x = self.fc_layers(x)
        return x

# 图像格式转换
convert_images_to_jpg(img_dir)

# 加载标签信息并初始化 MultiLabelBinarizer
styles_df = pd.read_csv(os.path.join(base_dir, 'tags.csv'), encoding='latin1')  # 确保正确的编码
styles = styles_df['tag'].tolist()  # 确保列名正确
mlb = MultiLabelBinarizer()
mlb.fit([[style] for style in styles])  # 每个标签作为一个单独的样本

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if torch.cuda.is_available() else 0
pin_memory = True if torch.cuda.is_available() else False

model = MultiLayerAttentionResNet(num_classes=len(styles))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image_styles(img_path, model, transform, label_encoder, threshold, device=device):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        return {}
    
    if transform:
        image = transform(image)
    
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)  # 使用sigmoid函数将输出转换为概率
        probabilities = probabilities.cpu().numpy().flatten()

    style_prob_dict = {label: prob for label, prob in zip(label_encoder.classes_, probabilities) if prob > threshold}

    sorted_style_prob_dict = dict(sorted(style_prob_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_style_prob_dict


# 加载模型
try:
    model.load_state_dict(torch.load(model_path))
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Failed to load model weights: {e}")

model = model.to(device)
model.eval()

for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        img_path = os.path.join(img_dir, filename)
        predictions = predict_image_styles(img_path, model, transform_val, mlb, threshold, device)
        print(f"Predictions for {filename}: {predictions}")

print("Classes:", mlb.classes_)