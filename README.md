
# **复旦大学医学影像分析24春季PJ2**


# 1. 简介
请参考我们的技术报告PPT

# 2. 运行

### **环境**

```
pip install -r requirements.txt
wandb off
```

## 训练


```python
python ./main.py --config_path=config/step1_pretraining/phase1.json
```


## 测试

测试cell count指标：

```python
python test.py
```

测试计算量参数量：

```python
python flops.py
```

## 模型下载

其中包含VM-UNet与MEDIAR-MAMBA的预训练模型，请将模型放在weights/pretrained文件夹下

- 链接：

[Baidu Drive Link](https://pan.baidu.com/s/1olRlmWP6LuXWem6zF4Q9zA?pwd=qy5o).

提取码：qy5o



