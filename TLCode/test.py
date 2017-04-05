import numpy as np
import tensorflow as tf
import  tensorlayer as tl
maxlen =30
max_features =20000
def load_imdb(path='imdb.npz', num_words=None, skip_top=0,
              maxlen=None, seed=113,
              start_char=1, oov_char=2, index_from=3, **kwargs):
    """Loads the IMDB dataset."""

    f = np.load(path)
    x_train = f['x_train']
    labels_train = f['y_train']
    x_test = f['x_test']
    labels_test = f['y_test']
    f.close()

    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(labels_train)

    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        new_xs = []
        new_labels = []
        for x, y in zip(xs, labels):
            if len(x) < maxlen:
                new_xs.append(x)
                new_labels.append(y)
        xs = new_xs
        labels = new_labels
    if not xs:
        raise ValueError('After filtering for sequences shorter than maxlen=' +
                         str(maxlen) + ', no sequence was kept. '
                         'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x] for x in xs]
    else:
        new_xs = []
        for x in xs:
            nx = []
            for w in x:
                if w >= num_words or w < skip_top:
                    nx.append(w)
            new_xs.append(nx)
        xs = new_xs

    x_train = np.array(xs[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])

    x_test = np.array(xs[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (x_train, y_train), (x_test, y_test)

def loadData():

    from keras.preprocessing import sequence
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = load_imdb(path='imdb.npz',num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("Pad sequences (samples x time)")
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.asarray(y_train,dtype='int32')
    y_test = np.asarray(y_test,dtype='int32')
    print  y_train.shape
    print x_train.shape
    return x_train,y_train,x_test,y_test