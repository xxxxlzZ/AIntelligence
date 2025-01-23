# 只输入图像
import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = 'lab5data/data/'
train_file = 'lab5data/train.txt'

train_data = pd.read_csv(train_file, sep=",", header=0, names=["guid", "label"])

images = []
labels = []
text_contents = []

for _, row in train_data.iterrows():
    guid = row['guid']
    label = row['label']

    # 图像路径
    image_path = os.path.join(data_folder, f"{guid}.jpg")
    if os.path.exists(image_path):
        images.append(image_path)
    else:
        images.append(None)

    # 文本路径及内容
    text_path = os.path.join(data_folder, f"{guid}.txt")
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='ISO-8859-1') as f:
            text_contents.append(f.read())
    else:
        text_contents.append(None)

    labels.append(label)

train_data['image_path'] = images
train_data['text_content'] = text_contents
train_data['label'] = labels

label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
train_data['label'] = train_data['label'].map(label_mapping)
train_data = train_data.dropna(subset=['text_content', 'label'])

tokenizer = BertTokenizer.from_pretrained("./model/bert-base-uncased")


def text_to_index(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt',
        truncation=True,
    )
    return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)


class ImageOnlyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 图像路径
        image_path = self.data_frame.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_indexes = torch.zeros((1, 128), dtype=torch.long)  # 假设长度与文本处理一致
        attention_mask = torch.zeros((1, 128), dtype=torch.long)  # 假设长度与文本处理一致

        label = self.data_frame.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)

        return image, text_indexes, attention_mask, label


class ImageOnlyModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ImageOnlyModel, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # 去掉最后一层
        self.fc = nn.Linear(2048, 3)  # 仅使用图像特征进行分类

    def forward(self, images):
        x_img = self.resnet(images).view(images.size(0), -1)
        return self.fc(x_img)


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
train_dataset = ImageOnlyDataset(train_data, transform=transform)
val_dataset = ImageOnlyDataset(val_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

image_only_model = ImageOnlyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(image_only_model.parameters(), lr=5e-5, weight_decay=1e-5)


def train_model(model, train_loader, val_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, texts, attention_masks, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, texts, attention_masks, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy on validation set (Image Only): {100 * correct / total:.2f}%')


train_model(image_only_model, train_loader, val_loader)