import os.path
import os
import time

import random
import numpy as np
import tensorflow as tf
import cv2
import scipy.ndimage
from sklearn.model_selection import train_test_split

import layer_def as ld
import BasicConvLSTMCell
import baseline.load_data

SAMPLE_RATE = 0.1
DATA_SAMPLE_RATE = 0.25

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .05,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_integer('label_size', 2,
                            """size of label vector""")
DATA_PATH = "sequence"

def load_lstm_data():
  data_map = baseline.load_data.get_data('data')
  examples = []
  labels = []
  for key in data_map:
      data_list = data_map[key]
      counter = 0
      data_label = data_list[0][2]
      if data_label:
          data_label_np = np.asarray([0,1])
      else:
          data_label_np = np.asarray([1,0])
      for i in range(len(data_list)):
          if data_label and random.uniform(0,1) > DATA_SAMPLE_RATE:
              continue
          broke_early = False
          point = []
          for j in range(i, i + FLAGS.seq_length):
              if j >= len(data_list):
                  broke_early = True
                  break
              index, path, label = data_list[j]
              point.append(path)
          if broke_early:
              break
          examples.append(point)
          labels.append(data_label_np)
  print len(examples)
  print len(labels)
  train_x, test_x, train_y, test_y = train_test_split(examples, labels, test_size = 0.2, stratify = labels)
  print len(train_x)
  print len(train_y)
  train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size = 0.25, stratify = train_y)
  return train_x, train_y, dev_x, dev_y, test_x, test_y

def load_frames(point, label):
  np_images = []
  for file_path in point:
    np_image = scipy.ndimage.imread(file_path)
    np_images.append(np_image)
  stacked = np.stack(np_images)
  return np.expand_dims(stacked,axis=0), np.asarray([label])

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

def network(inputs, hidden):
  conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
  # conv3
  conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
  y_0 = conv4
  with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([27,48], [3,3], 4)
    #cell = BasicConvLSTMCell.BasicConvLSTMCell([8,8], [3,3], 4)
    if hidden is None:
      hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
    y_1, hidden = cell(y_0, hidden)
  reshaped = tf.reshape(y_0, [1, 5184])
  fully_connected_output = tf.layers.dense(reshaped, 2)
  return fully_connected_output, hidden, conv4

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 108, 192, 3])
    y = tf.placeholder(tf.float32, [1, FLAGS.label_size])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []

    # conv network
    hidden = None

    '''for i in range(FLAGS.seq_length):
      x_1, hidden, re = network_template(x_dropout[:,i,:,:,:], hidden)
      x_unwrap.append(x_1)
    '''
    x_1, hidden, re = network_template(x_dropout[:,0,:,:,:], hidden)
    x_unwrap.append(x_1)

    # pack them all together 
    x_unwrap = tf.stack(x_unwrap)

    # calc total loss
    #print(x_unwrap)
    #predicted_task_vector = x_unwrap[FLAGS.seq_length-1,:,:]
    predicted_task_vector = x_unwrap[0,:,:]
    loss_vector = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predicted_task_vector)
    loss = tf.reduce_mean(loss_vector)
    tf.summary.scalar('loss', loss)
    correct_pred = tf.equal(tf.argmax(predicted_task_vector, 1), tf.argmax(y, 1))
   # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    weights1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    debug_grad = tf.gradients(ys=re, xs = weights1[:-4])
    #print(debug_grad)
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    x_train, y_train, x_dev, y_dev, x_test, y_test = load_lstm_data()

    for step in xrange(FLAGS.max_step):
      #index = step % len(x_train)
      index = random.randint(0, len(x_train) - 1)
      features, labels = load_frames(x_train[index], y_train[index])
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        
        #print sess.run(debug_grad, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
        summary_str = sess.run(summary_op, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
        summary_writer.add_summary(summary_str, step) 
        #print("time per batch is " + str(elapsed))
        print(step)
        print("loss for this image was " + str(loss_r))
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%2000 == 1999:
        total = 0
        false_pos = 0
        false_neg = 0
        true_pos = 0
        true_neg = 0
        avg_loss = 0.0
        for i in range(len(x_dev)):
            if random.uniform(0, 1) > SAMPLE_RATE:
                continue
            total += 1
            dev_feature, dev_label = load_frames(x_dev[i], y_dev[i])
            accuracy = sess.run(correct_pred, feed_dict={x:dev_feature, y:dev_label, keep_prob:FLAGS.keep_prob})
            loss_r = sess.run(loss,feed_dict={x:dev_feature, y:dev_label, keep_prob:FLAGS.keep_prob})
            avg_loss += loss_r
            #print accuracy, y_dev[i]
            if accuracy[0]:
                if y_dev[i][1]:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if y_dev[i][1]:
                    false_neg += 1
                else:
                    false_pos += 1
        avg_loss /= len(x_dev)
        print ('total: ' + str(total))
        print ('true_pos: ' + str(true_pos))
        print ('true_neg: ' + str(true_neg))
        print ('false_pos: ' + str(false_pos))
        print ('false_neg: ' + str(false_neg))
        print("accuracy: " + str(float(true_pos + true_neg) / total))
        print('avg_loss: ' + str(avg_loss))
        print

        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

    print 'FINAL TEST METRICS'
    total = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    avg_loss = 0.0
    for i in range(len(x_test)):
        #if random.uniform(0, 1) > SAMPLE_RATE:
        #    continue
        total += 1
        test_feature, test_label = load_frames(x_test[i], y_test[i])
        accuracy = sess.run(correct_pred, feed_dict={x:test_feature, y:test_label, keep_prob:FLAGS.keep_prob})
        loss_r = sess.run(loss,feed_dict={x:test_feature, y:test_label, keep_prob:FLAGS.keep_prob})
        avg_loss += loss_r

        if accuracy[0]:
            if y_test[i][1]:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if y_test[i][1]:
                false_neg += 1
            else:
                false_pos += 1
    avg_loss /= len(x_test)
    print ('total: ' + str(total))
    print ('true_pos: ' + str(true_pos))
    print ('true_neg: ' + str(true_neg))
    print ('false_pos: ' + str(false_pos))
    print ('false_neg: ' + str(false_neg))
    print("accuracy: " + str(float(true_pos + true_neg) / total))
    print('avg_loss: ' + str(avg_loss))
    print


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


