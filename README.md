# InfoDSGAN

Train models

```
python train.py
```

Run experiments

```
pyhon test.py
```

Code implementation of infoNCE
```python
import tensorflow as tf
from tensorflow.keras import losses

ce = losses.SparseCategoricalCrossentropy(from_logits=True)
def infoNCE(f_q, f_k, tau=0.07, y=None):
    '''
    f_q: (b,d)  from  Q(Enc_s(Dec(c,shift(s))))
    f_k: (b,d), from  shift(Q(s))
    '''
    b, d = f_q.shape

    f_q = tf.math.l2_normalize(f_q, axis=-1)
    f_k = tf.math.l2_normalize(f_k, axis=-1)

    # positive
    l_pos = tf.reduce_sum(f_q * f_k, axis=-1, keepdims=True)  # (b,1)

    # negative
    if y is None:
        y = tf.eye(b)
    else:
        y = tf.one_hot(y, tf.reduce_max(y) + 1, axis=-1)
        y = y @ tf.transpose(y)
        
    mask = tf.where(y==1, -float('inf'), y)
    l_neg = f_q @ tf.transpose(f_k)  # (b,b)
    l_neg = mask + l_neg
    
    # compute loss
    logits = tf.concat([l_pos, l_neg], axis=-1) / tau  # (b,b+1)
    targets = tf.zeros((b,))
    loss = ce(targets, logits)
    # loss=tf.reduce_mean(
    # -tf.math.log(tf.exp(logits[:,0])/\
    # tf.reduce_sum(tf.exp(logits),axis=-1)))
    return loss
```

Code implementation of infoDCE
```python
import tensorflow as tf
from tensorflow.keras import losses

ce = losses.SparseCategoricalCrossentropy(from_logits=True)
def infoDCE(f, e, y, tau=0.07):
    b, d = f.shape
    f = tf.math.l2_normalize(f, axis=-1)
    e = tf.math.l2_normalize(e, axis=-1)
    l_pos = tf.reduce_sum(f * e, axis=-1, keepdims=True)  # (b,1)
    
    y = tf.cast(y, 'int32')
    y = tf.one_hot(y, tf.reduce_max(y) + 1, axis=-1)
    y = y @ tf.transpose(y)
    mask = tf.where(y==1, -float('inf'), y)
    l_neg = f @ tf.transpose(f) + mask
    logits = tf.concat([l_pos, l_neg], axis=-1) / tau  # (b,b+1)
    targets = tf.zeros((b,))
    loss = ce(targets, logits)
    return loss, \
           tf.reduce_mean(l_pos), \
           tf.reduce_mean(tf.math.log(tf.exp(l_neg) + y))
```
