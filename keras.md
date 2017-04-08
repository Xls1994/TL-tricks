# Keras使用技巧  
Keras是一个高度封装的深度学习框架，支持在theano和tensorflow为后端的系统下运行。Keras为用户提供了大量简单而实用的api函数，
本文试图总结在keras使用中的技巧和问题。如果你发现了一些keras使用的陷阱或者技巧，请联系我们。在确认之后，我们会添加到这里。

## F&QA  
1.Q：网上下载的代码为什么出现`import`错误？<br>
A：首先检查你的keras版本
```bash
  >>>import keras
  >>>keras.__version__
```
如果你的版本是2.0以上，那么1.X版本的代码出现`import`问题时，很有可能是api函数已经改变位置，或者不再使用。请到Keras官网查看最新api<br>
A2：同步你的后端，如果你使用的是2.0版本的Keras，后端选用Tensorflow。那么请更新Tensorflow到r1.0版本,这样保证函数的一致性。<br>
2.卷积操作得到的图像大小？
* 输入图片大小 W×W
* Filter大小 F×F
* 步长 S
* padding的像素数 P <br>
于是我们可以得出
```
N = (W − F + 2P )/S+1
```
##  常用链接
 * Keras官方网站: [Docs](https://keras.io/), [中文文档](http://keras-cn.readthedocs.io/en/latest/), [Github](https://github.com/fchollet/keras)
 * [常用Keras模型 ](https://github.com/fchollet/deep-learning-models)
 * [Keras教程](https://github.com/leriomaggio/deep-learning-keras-tensorflow)
 * Follow [fchollet](https://github.com/fchollet/keras) for further examples
