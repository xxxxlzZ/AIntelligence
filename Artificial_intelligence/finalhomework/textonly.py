# 只输入文本
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

data_folder = 'lab5data/lab5data/'
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


class TextOnlyDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 文本内容
        text_content = self.data_frame.iloc[idx]['text_content']
        text_indexes, attention_mask = text_to_index(text_content)

        label = self.data_frame.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)

        # 所有图像数据都设为0
        image = torch.zeros((3, 128, 128))
        return image, text_indexes, attention_mask, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
train_dataset = TextOnlyDataset(train_data, transform=transform)
val_dataset = TextOnlyDataset(val_data, transform=transform)


def collate_fn(batch):
    images, texts, attention_masks, labels = zip(*batch)
    images = torch.stack(images, 0)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return images, texts, attention_masks, labels


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class TextOnlyModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(TextOnlyModel, self).__init__()
        self.bert_model = BertModel.from_pretrained('./model/bert-base-uncased')
        self.fc = nn.Linear(768, 3)

    def forward(self, text_indexes, attention_mask):
        outputs = self.bert_model(input_ids=text_indexes, attention_mask=attention_mask)
        x_text = outputs.last_hidden_state.mean(dim=1)
        return self.fc(x_text)


text_only_model = TextOnlyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(text_only_model.parameters(), lr=5e-5, weight_decay=1e-5)


def train_model(model, train_loader, val_loader, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, texts, attention_masks, labels in train_loader:
            images, texts, attention_masks, labels = images.to(device), texts.to(device), attention_masks.to(
                device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts, attention_masks)
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
            images, texts, attention_masks, labels = images.to(device), texts.to(device), attention_masks.to(
                device), labels.to(device)
            outputs = model(texts, attention_masks)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy on validation set (Text Only): {100 * correct / total:.2f}%')


train_model(text_only_model, train_loader, val_loader)