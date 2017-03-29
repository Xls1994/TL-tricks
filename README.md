# 如何使用Tensorlayer  
随着深度学习在全世界的普及，我们必须学会使用一些技巧来实现深度学习的算法。Tensorlayer是一个基于tensorflow的高层api，使用tensorlayer可以让我们更好的搭建自己的深度学习模型。

这里有一些关于使用Tensorlayer的技巧，当然你也可以在[FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#fqa)发现更多的技巧.假如你在实践中发现里一些有用的小技巧，请pull一下我们。如果我们发现这个技巧是合理的，经过确认我们会把它总结在这里。

## 1. 安装
 * 为了使你的TL保持最新的版本，并能够轻易的修改源代码，你可以通过以下命令下载整个tl项目 `git clone https://github.com/zsdonghao/tensorlayer.git`，然后把整个`tensorlayer`文件夹放在你的项目里 
 * TL更新的速度十分频繁，如果你使用`pip`安装，我们建议你安装master版本  
 * 如果你要进行自然语言处理的相关操作，我们建议你安装 [NLTK and NLTK data](http://www.nltk.org/install.html)

## 2. TF 和 TL之间的衔接
 * TF to TL : use [InputLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#input-layer)
 * TL to TF : use [network.outputs](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
 * 其他方法 [issues7](https://github.com/zsdonghao/tensorlayer/issues/7), 多输入问题 [issues31](https://github.com/zsdonghao/tensorlayer/issues/31)

## 3. 训练/测试切换
 * 使用 [network.all_drop](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer) 控制训练和测试不同阶段的dropout (for [DropoutLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dropout-layer) only) see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py) and [Understand Basic layer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
 * 或者, 把 `is_fix` 设置为 `True` in [DropoutLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dropout-layer), 并通过使用不同的参数，为训练和测试构建不同的图. 你也可以设置不同的`batch_size` 噪音概率来创建不同的图。这个方法在你使用[GaussianNoiseLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#gaussian-noise-layer), [BatchNormLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#batch-normalization) 等，是十分有效的手段. 这里有一个简单的例子:
```python
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      net = InputLayer(x, name='in')
      net = DropoutLayer(net, 0.8, True, is_train, 'drop1')
      net = DenseLayer(net, 800, tf.nn.relu, 'dense1')
      net = DropoutLayer(net, 0.8, True, is_train, 'drop2')
      net = DenseLayer(net, 800, tf.nn.relu, 'dense2')
      net = DropoutLayer(net, 0.8, True, is_train, 'drop3')
      net = DenseLayer(net, 10, tf.identity, 'out')
      logits = net.outputs
      net.outputs = tf.nn.sigmoid(net.outputs)
      return net, logits
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
net_train, logits = mlp(x, is_train=True, reuse=False)
net_test, _ = mlp(x, is_train=False, reuse=True)
cost = tl.cost.cross_entropy(logits, y_, name='cost')
```


## 4. 获取训练的变量
 * 使用 [tl.layers.get_variables_with_name](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#get-variables-with-name) 来代替 [net.all_params](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
```python
train_vars = tl.layers.get_variables_with_name('MLP', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_vars)
```
 * 这个方法可以用来在训练时冻结某些层，只要简单的不获取这些变量
 * 其他方法 [issues17](https://github.com/zsdonghao/tensorlayer/issues/17), [issues26](https://github.com/zsdonghao/tensorlayer/issues/26), [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#exclude-some-layers-from-training)
  
## 5. 预训练的 CNN 和 Resnet
* 预训练的 CNN
  * 许多应用需要预训练的CNN模型
  * TL 的例子里提供了预训练的 VGG16, VGG19, Inception and etc : [TL/example](https://github.com/zsdonghao/tensorlayer/tree/master/example)
  * [tl.layers.SlimNetsLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-tf-slim) 允许你使用所有 [Tf-Slim pre-trained models](https://github.com/tensorflow/models/tree/master/slim)
* Resnet
  * 利用"for"实现 [issues85](https://github.com/zsdonghao/tensorlayer/issues/85)
  * 其他 [by @ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer)

## 6. 数据增强
* Use TFRecord, see [cifar10 and tfrecord examples](https://github.com/zsdonghao/tensorlayer/tree/master/example); good wrapper: [imageflow](https://github.com/HamedMP/ImageFlow)
* Use python-threading with [tl.prepro.threading_data](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#threading) and [the functions for images augmentation](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#images) see [tutorial_image_preprocess.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_image_preprocess.py)

## 7. 批量数据使用
* 如果你的数据足够小，能够加载到内存里
  * 使用 [tl.iterate.minibatches](http://tensorlayer.readthedocs.io/en/latest/modules/iterate.html#tensorlayer.iterate.minibatches) 来返回打乱顺序的batch大小的数据和标签.
  * 时间序列的数据可以使用 [tl.iterate.seq_minibatches, tl.iterate.seq_minibatches2, tl.iterate.ptb_iterator and etc](http://tensorlayer.readthedocs.io/en/latest/modules/iterate.html#time-series)
* 如果你的数据十分庞大
  * 使用 [tl.prepro.threading_data](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#tensorlayer.prepro.threading_data) to read a batch of data at the beginning of every step
  * 使用 TFRecord，请参考  [cifar10 and tfrecord examples](https://github.com/zsdonghao/tensorlayer/tree/master/example)


## 8. 句子切分
 * 使用 [tl.nlp.process_sentence](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#process-sentence)来切分你的句子 ,这个函数需要 [NLTK and NLTK data](http://www.nltk.org/install.html)
 * 然后使用 [tl.nlp.create_vocab](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#create-vocabulary) 创建一个词典，保存为txt文件(它会返回一个[tl.nlp.SimpleVocabulary object](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#simple-vocabulary-class)让词和词的编号进行对应)
 * 最后使用 [tl.nlp.Vocabulary](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#vocabulary-class) 在 `tl.nlp.create_vocab`创建的txt文件中创建一个字典对象
 * 更多关于句子预处理的函数请查看 [tl.prepro](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#sequence) 和 [tl.nlp](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html)

## 9. 动态RNN和句子长度
 * 使用 [tl.layers.retrieve_seq_length_op2](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#compute-sequence-length-2) 自动从placeholder中计算句子的长度,然后把他输入到 `sequence_length`  [DynamicRNNLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dynamic-rnn-layer)
 * 给batch里的数据添加一共0的填充，可以使用下面这个例子:
```python
b_sentence_ids = tl.prepro.pad_sequences(b_sentence_ids, padding='post')
```
 * 其他方法 [issues18](https://github.com/zsdonghao/tensorlayer/issues/18)

## 10. 共性的问题
 * 导入Tensorlayer时Matplotlib出现问题 [issues](https://github.com/zsdonghao/tensorlayer/issues/79), [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#visualization)
 
## 11. 其他技巧
 * 取消控制台打印: 如果你正在构建一个非常深的神经网络，并不想在控制台看到相关的信息d。你可以使用 tl.ops.suppress_stdout():`:
```
print("You can see me")
with tl.ops.suppress_stdout():
    print("You can't see me") # build your graphs here
print("You can see me")
```
## 12. 使用其他的TF包装器
TL可以和其他TF的包装器在一起使用，如果你使用其他的API编写的代码，或者别人提供的代码，你可以轻而易举的使用它 !
 * Keras to TL: [KerasLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-keras) (if you find some codes implemented by Keras, just use it. example [here](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py))
 * TF-Slim to TL: [SlimNetsLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-tf-slim) (you can use all Google's pre-trained convolutional models with this layer !!!)
 * 将来应该会适配更多的高层API
## 13. 不同版本TF的适用
 * [RNN cell_fn](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html): 使用 [tf.contrib.rnn.{cell_fn}](https://www.tensorflow.org/api_docs/python/) for TF1.0+, 或者 [tf.nn.rnn_cell.{cell_fn}](https://www.tensorflow.org/versions/r0.11/api_docs/python/) for TF1.0-
 * [cross_entropy](http://tensorlayer.readthedocs.io/en/latest/modules/cost.html): TF1.0+必须定义一共独特的名字
 
## 有用的链接
 * TL官方网站: [Docs](http://tensorlayer.readthedocs.io/en/latest/), [中文文档](http://tensorlayercn.readthedocs.io/zh/latest/), [Github](https://github.com/zsdonghao/tensorlayer)
 * [Learning Deep Learning with TF and TL ](https://github.com/wagamamaz/tensorflow-tutorial)
 * Follow [zsdonghao](https://github.com/zsdonghao) for further examples

## 作者
 - Zhang Rui
 - You
 - Icy 翻译
