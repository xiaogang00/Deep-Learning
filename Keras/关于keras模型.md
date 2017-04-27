Keras有两种类型的模型，序贯模型(Sequential)和函数式模型(Model)，函数式模型应用更为广泛，序贯模型是函数式模型的一种特殊情况。

两类模型的一些方法是相同的：

model.summary():  打印出模型概况

model.get_config()：返回包含模型配置信息的python字典

```
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config
```

- `model.get_layer()`：依据层名或下标获得层对象
- `model.get_weights()`：返回模型权重张量的列表，类型为numpy array
- `model.set_weights()`：从numpy array里将权重载入给模型，要求数组具有与`model.get_weights()`相同的形状。
- `model.to_json`：返回代表模型的JSON字符串，仅包含网络结构，不包含权值。可以从JSON字符串中重构原模型：


- `model.save_weights(filepath)`：将模型权重保存到指定路径，文件类型是HDF5（后缀是.h5）
- `model.load_weights(filepath, by_name=False)`：从HDF5文件中加载权重到当前模型中, 默认情况下模型的结构将保持不变。如果想将权重载入不同的模型（有些层相同）中，则设置`by_name=True`，只有名字匹配的层才会载入权重