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
from torchvision import models
from tqdm import tqdm
import warnings

# 这是忽略特定警告使得没有多余的输出
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 如果能用GPU就用GPU，加速运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = 'lab5data/lab5data/'
train_file = 'lab5data/train.txt'
test_file = 'lab5data/test_without_label.txt'

# 读取训练数据和训练标签
train_data = pd.read_csv(train_file, sep=",", header=0, names=["guid", "label"])

images = []
labels = []
text_contents = []

# 读取guid对应的图片和文本，还有情感标签
for _, row in train_data.iterrows():
    guid = row['guid']
    label = row['label']

    image_path = os.path.join(data_folder, f"{guid}.jpg")
    text_path = os.path.join(data_folder, f"{guid}.txt")

    if os.path.exists(image_path):
        images.append(image_path)
    else:
        images.append(None)

    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='ISO-8859-1') as f:
            text_contents.append(f.read())
    else:
        text_contents.append(None)

    labels.append(label)

# 将上面读到的加入train_data中
train_data['image_path'] = images
train_data['text_content'] = text_contents
train_data['label'] = labels

# 将情感标签从字符串转换为数字
label_mapping = {"positive": 0, "neutral": 1, "negative": 2}
train_data['label'] = train_data['label'].map(label_mapping)

# 看是否有空的标签
assert train_data['label'].notnull().all(), "There are NaN labels after mapping."
train_data = train_data.dropna(subset=['image_path', 'text_content', 'label'])

# 初始化调用的BERT模型
tokenizer = BertTokenizer.from_pretrained("./model/bert-base-uncased")


# 将输入的文本字符串转换为 BERT 模型可以接受的输入格式
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


# 定义数据集类
class MultiModalDataset(Dataset):

    # 初始化数据集
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform

    # 返回数据集的长度
    def __len__(self):
        return len(self.data_frame)

    # 这里是用于获取特定样本的数据, idx: 该参数表示索引
    def __getitem__(self, idx):
        image_path = self.data_frame.iloc[idx]['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        text_content = self.data_frame.iloc[idx]['text_content']
        text_indexes, attention_mask = text_to_index(text_content)

        label = self.data_frame.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)

        return image, text_indexes, attention_mask, label


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
])

train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
train_dataset = MultiModalDataset(train_data, transform=transform)
val_dataset = MultiModalDataset(val_data, transform=transform)


# 将从 MultiModalDataset 提取的一组样本组合成统一的批次格式
def collate_fn(batch):
    images, texts, attention_masks, labels = zip(*batch)
    images = torch.stack(images, 0)
    # 比如将文本张量的长度统一
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return images, texts, attention_masks, labels


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


class CrossAttentionLayer(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim):
        super(CrossAttentionLayer, self).__init__()
        self.text_transform = nn.Linear(text_dim, hidden_dim)
        self.image_transform = nn.Linear(image_dim, hidden_dim)

    def forward(self, text_feats, image_feats):
        text_transformed = self.text_transform(text_feats)
        image_transformed = self.image_transform(image_feats)

        # image_transformed 的维度应为 (batch_size, 1, hidden_dim)
        image_transformed = image_transformed.unsqueeze(1)

        scores = torch.bmm(text_transformed, image_transformed.transpose(1, 2))
        attn_weights = torch.softmax(scores, dim=1)
        context_vector = torch.bmm(attn_weights.transpose(1, 2), text_transformed)
        context_vector = context_vector.squeeze(1)

        return context_vector


# 构建的多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MultiModalModel, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.bert_model = BertModel.from_pretrained('./model/bert-base-uncased')

        # 跨模态注意力机制
        self.cross_attention = CrossAttentionLayer(text_dim=768, image_dim=2048, hidden_dim=128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(128 + 768, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, images, text_indexes, attention_mask):
        x_img = self.resnet(images).view(images.size(0), -1)
        outputs = self.bert_model(input_ids=text_indexes, attention_mask=attention_mask)
        x_text = outputs.last_hidden_state
        context_vector = self.cross_attention(x_text, x_img)
        x_combined = torch.cat((context_vector, x_text.mean(dim=1)), dim=1)

        x = torch.relu(self.fc1(x_combined))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = MultiModalModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]")
    for images, texts, attention_masks, labels in loop:
        images, texts, attention_masks, labels = images.to(device), texts.to(device), attention_masks.to(
            device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, texts, attention_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    scheduler.step()

# 验证模型的准确率
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, texts, attention_masks, labels in val_loader:
        images, texts, attention_masks, labels = images.to(device), texts.to(device), attention_masks.to(
            device), labels.to(device)
        outputs = model(images, texts, attention_masks)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on validation set: {100 * correct / total:.2f}%')


# 预测测试集
def predict_test_set(test_file):
    # 读取测试集文件
    test_data = pd.read_csv(test_file, sep=",", header=0)
    predictions = []
    test_data['tag'] = 'null'  # 初始化标签的空值

    # 不同标签的映射
    label_mapping_inv = {0: "positive", 1: "neutral", 2: "negative"}

    for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Predicting"):
        guid = row['guid']

        image_path = os.path.join(data_folder, f"{guid}.jpg")
        text_path = os.path.join(data_folder, f"{guid}.txt")

        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)  # 转到设备并添加batch维度
        else:
            print(f"Image for GUID {guid} not found.")
            continue

        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='ISO-8859-1') as f:
                text_content = f.read()
        else:
            print(f"Text for GUID {guid} not found.")
            continue

        text_indexes, attention_mask = text_to_index(text_content)
        text_indexes = text_indexes.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        # 进行预测
        with torch.no_grad():
            output = model(image, text_indexes, attention_mask)
            _, predicted = torch.max(output.data, 1)


        predictions.append(label_mapping_inv[predicted.item()])

    test_data['tag'] = predictions

    # 将预测结果写入文件
    output_file = 'lab5data/test_with_predictions.txt'
    test_data.to_csv(output_file, index=False)

    print(f'Predictions saved to {output_file}')

# 调用预测函数
predict_test_set(test_file)