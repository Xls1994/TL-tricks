import numpy as np
import tensorflow as tf
import tensorlayer as tl
from  test  import loadData
maxlen=30
max_features =20000
embedding_size =128
batch_size =128



x_train,y_train,x_test,y_test =loadData()
print np.max(x_train)
print np.shape(y_train)
sess =tf.InteractiveSession()
x = tf.placeholder(tf.int32, shape=[None, maxlen], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
def keras_blok(x):
    from keras.layers import LSTM,Dropout,Embedding
    x =Embedding(max_features,128)(x)
    x =LSTM(50)(x)
    x =Dropout(0.5)(x)
    return x


network = tl.layers.EmbeddingInputlayer(x, vocabulary_size=max_features, embedding_size=embedding_size)
# network=tl.layers.KerasLayer(leftnetwork,keras_layer=keras_blok,name='keras')
# print network.outputs.shape
network =tl.layers.RNNLayer(network,cell_fn=tf.contrib.rnn.BasicLSTMCell,
n_hidden=100,n_steps=30,return_last=True,return_seq_2d=True
                            )
network = tl.layers.DenseLayer(network, n_units=2,
                                   act=tf.identity,
                                   name='output_layer')

y = network.outputs

#cost function
cost = tl.cost.cross_entropy(y, y_,name='cost1')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(y, 1)
#train op Optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# sess.run(tf.global_variables_initializer())
tl.layers.initialize_global_variables(sess)
#print params and layers
network.print_params()
network.print_layers()
def trainNetwork():
    from keras import backend as K
    import time
    n_epoch =20
    for epoch in range(n_epoch):
        start_time =time.time()
        for X_train_a,y_train_a in tl.iterate.minibatches(
            x_train,y_train,batch_size=batch_size,
            shuffle=True):
            feed_dicts ={x:X_train_a,y_:y_train_a,K.learning_phase():1}
            feed_dicts.update(network.all_drop)
            _,_ =sess.run([cost,train_op],feed_dict=feed_dicts
                          )
        print 'Epoch %d of %d took %fs' %(epoch+1,n_epoch,time.time()-start_time)
        train_loss,train_acc,n_batch =0,0,0
        for X_train_a,y_train_a in tl.iterate.minibatches(
            x_train,y_train,batch_size,shuffle=False
        ):
            feed_dicts ={x:X_train_a,y_:y_train_a,K.learning_phase():0}
            # feed_dicts.update(network.all_drop)
            err,ac =sess.run([cost,acc],feed_dict=feed_dicts)
            train_loss += err
            train_acc+=ac
            n_batch +=1
        print 'train loss: %f' %(train_loss/n_batch)
        print 'train acc: %f' %(train_acc/n_batch)

trainNetwork()
# tl.utils.fit(sess,network,train_op,cost,x_train,y_train,x,y_,n_epoch=10,batch_size=20)
tl.files.save_npz()
sess.close()