使用TensorFlow的`queue`来优化这个程序，和TensorFlow中的其他对象一样，queue也只是TensorFlow中的一个节点，其可以将feed_dict操作转化为异步计算。
从`tensorflow.python.ops.data_flow_ops`的源码中可以看到，所有的`Queue`类都是继承自`QueueBase`父类，继承自QueueBase类的子类有：

* RandomShuffleQueue 随机队列
* FIFOQueue 先入先出
* PaddingFIFOQueue 可以自带padding
* PriorityQueue 优先队列

并且每个类都有以下几个常用的方法：

* enqueue() 即向queue中压入数据
* dequeue() 即从queue中弹出数据
* enqueue_many() 即向queue中压入多个数据
* dequeue_many() 即从queue中弹出多个数据

当然，单个这个Queue类的方法好像并没有什么吸引人之处，因为就它自身而言，如果我们启动了一个queue，它是生产者消费者模型.
如果只是在单一线程下面工作，那仍然是无济于事的，就好比很多个厨师一起做菜，然而却只有一个灶台可以利用。
因此，要想提高运行效率，我们必须要让enqueue和dequeue分别处在不同线程上，这个时候就需要用到QueueRunner类和Coordinator类了。

`QueueRunner`类和`Coordinator`类的作用是处理队列的操作、保持同步，并且这些操作都是在不同的线程上面。
根据`tensorflow.python.training`中的源码可知，QueueRunner需要一个队列和入队的操作（可以是很多个操作），然后根据session即可创造出很多个执行入队操作的线程，然后调用`tf.train.add_queue_runner`方法即可将`queue_runner`添加到`TensorFlow QUEUE_RUNNERS`集合中去，再调用tf.train.start_queue_runner方法即可启动所有线程。
这样，就可以调用上层API Coordinate来执行线程了。

Coordinator类可以让多个线程停止，它主要有三个方法：

* tf.train.Coordinator.should_stop 确认线程是否应该停止
* tf.train.Coordinator.request_stop 要求线程停止
* tf.train.Coordinator.join 要求等待线程结束

Coordinator可以为线程做很多事情，还可以捕捉到线程的异常以及报告错误，这里我不详细说明如何调用这些。
现在将上面那个BLSTM的程序该写成使用FIFOQueue和Coordinator来执行：
```python
import time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

time_length = 128
batch_size = 400
feature_size = 512

hidden_size=128

x = tf.random_normal([time_length, batch_size, feature_size], mean=0, stddev=1)## prepare dataq = tf.FIFOQueue(capacity=4, dtypes=tf.float32) 
enqueue_op = q.enqueue(x)
num_threads = 1 qr = tf.train.QueueRunner(q, [enqueue_op] * num_threads)
tf.train.add_queue_runner(qr)
inputs = q.dequeue() 
inputs.set_shape(x.get_shape())
y = tf.reduce_mean(tf.reduce_sum(inputs, axis=0), axis=1, keep_dims=True)
labels = tf.cast(tf.greater(y, 0), tf.int32)

## build model
sequence_length = tf.Variable([time_length]*batch_size, dtype=tf.int32)
cell_fw = LSTMCell(num_units=hidden_size)
cell_bw = LSTMCell(num_units=hidden_size)
outputs, state = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs, 
      sequence_length=sequence_length,
      dtype=tf.float32,
      time_major=True)

outputs_fw, outputs_bw = outputs
outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
outputs = tf.reduce_mean(outputs, axis=0)
outputs = tf.contrib.layers.fully_connected(
            inputs=outputs,
            num_outputs=1,
            activation_fn=None)

losses_op = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(labels, tf.float32), outputs)
losses_op = tf.reduce_mean(losses_op)

y_pred = tf.cast(tf.greater(outputs, 0), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, labels), \
  tf.float32))
adam = tf.train.AdamOptimizer(0.001)
train_op = adam.minimize(losses_op, name="train_op")

t1=time.time()with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  for i in range(100):
    _, losses, acc = sess.run([train_op, losses_op, accuracy])
    print 'epoch:%d, loss: %f'%(i, losses)

  coord.request_stop()
  coord.join(threads)
  print("Time taken: %f" % (time.time() - t1))
  ```
  
