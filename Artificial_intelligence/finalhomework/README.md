## Setup
torch>=1.7.0

torchvision>=0.8.0

transformers>=4.0.0

pandas>=1.1.0

Pillow>=8.0

scikit-learn>=0.24.0

tqdm>=4.42.0

You can simply run

```python
pip install -r requirements.txt
```

## Repository structure

因为大文件无法上传，我会上传压缩包到网盘里，麻烦老师手动解压一下，对应文件夹下会有README提示
下面是网盘连接 
链接：https://pan.quark.cn/s/a007f6b20e97

```python
|-- lab5data # 实验数据
    |-- test_with_predictions.txt # 预测结果文件
    |-- data/  # 数据集
    |-- train.txt  # 训练姐
    |-- test_without_label.txt # 测试集
|-- model # 预训练的模型
    |-- bert-base-uncased/ 
    |-- resnet50/ 
|-- emotionalmodel.py # 多模态模型
|-- imageonly.py # 只输入图片
|-- textonly.py # 只输入文本
```
## Run

```python
python emotionalmodel.py 
```
