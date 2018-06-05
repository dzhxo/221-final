import os.path
import os
import time

import h5py
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
tf.app.flags.DEFINE_integer('max_step', 10000,
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

#h5_file = h5py.File('images_processed.h5', 'r')

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
#  print train_x
  return train_x, train_y, dev_x, dev_y, test_x, test_y

def load_frames(points, labels):
  all_x = []
  for point, label in zip(points, labels):
    np_images = []
    for file_path in point:
      np_image = scipy.ndimage.imread(file_path)
      np_images.append(np_image)
    all_x.append(np_images)
  stacked = np.stack(all_x)
  return np.squeeze(np.expand_dims(all_x,axis=0)), np.asarray(labels)

FEATURE_PATH = 'data_featurized'

def load_features(points, labels):
  all_x = []
  for point, label in zip(points, labels):
    np_features = []
    for file_path in point:
      new_name = os.path.splitext(os.path.split(file_path)[1])[0] + '.txt'
      feature_path = os.path.join(FEATURE_PATH, new_name)
#      print feature_path
      features = np.fromfile(feature_path, sep=',')
#      print features
      np_features.append(features)
    all_x.append(np_features)
  stacked = np.stack(all_x)
  return np.squeeze(np.expand_dims(all_x,axis=0)), np.asarray(labels)
  

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 

def network(inputs, hidden):
  conv1 = ld.conv_layer(inputs, 3, 2, 8, "encode_1")
  # conv2
  conv2 = ld.conv_layer(conv1, 3, 1, 8, "encode_2")
  # conv3
  conv3 = ld.conv_layer(conv2, 3, 2, 8, "encode_3")
  # conv4
  conv4 = ld.conv_layer(conv3, 1, 1, 4, "encode_4")
  #with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
  #  cell = BasicConvLSTMCell.BasicConvLSTMCell([27,48], [3,3], 4)
  #  #cell = BasicConvLSTMCell.BasicConvLSTMCell([8,8], [3,3], 4)
  #  if hidden is None:
  #    hidden = cell.zero_state(FLAGS.batch_size, tf.float32) 
  #  y_1, hidden = cell(conv4, hidden)
  return conv4

# make a template for reuse
network_template = tf.make_template('network', network)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 1280])
    y = tf.placeholder(tf.float32, [None, FLAGS.label_size])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)
    # create network
    #x_unwrap = []

    # conv network
    #hidden = None
    #for i in range(FLAGS.seq_length):
    #  x_1 = network_template(x_dropout[:,i,:], hidden)
    #  x_unwrap.append(x_1)


    # pack them all together 
    #x_unwrap = tf.stack(x_unwrap)


    # calc total loss

    #predicted_task_vector = x_unwrap[FLAGS.seq_length-1,:,:]
    #end_states = x_unwrap[FLAGS.seq_length - 1,:, :,:]
    #transposed = tf.transpose(x_unwrap, [1,0,2])
    rnn_cell = tf.contrib.rnn.BasicRNNCell(2)
    initial_state = rnn_cell.zero_state(16, dtype=tf.float32)
    #transposed = tf.transpose(x_dropout, [1,0,2])
    #reshaped1 = tf.reshape(transposed, [16, 10, 5184]) 
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, x_dropout, initial_state=initial_state)#`, dtype=tf.float32)
    end_states = outputs[:,FLAGS.seq_length-1,:]
    #fully_connected_output = tf.layers.dense(end_states, units=2)
    
    loss_vector = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=end_states)
    loss = tf.reduce_mean(loss_vector)
    tf.summary.scalar('loss', loss)
    correct_pred = tf.equal(tf.argmax(end_states, 1), tf.argmax(y, 1))
    tf_accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
    probs = tf.nn.softmax(end_states)   
# training
    tvars = tf.trainable_variables()
    print tvars
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(FLAGS.lr)
    weights1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    #debug_grad = tf.gradients(ys=re, xs = weights1[:-4])
    #print(debug_grad)
    # List of all Variables
    variables = tf.global_variables()
   #grads = tf.gradients(loss, tvars)
    #train_op = optimizer.apply_gradients(zip(grads, tvars))
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
      index = (step * 16) % len(x_train)
      if index + 16 > len(x_train):
        continue  
	#index = (step * 16) % len(x_train)
      #index = random.randint(0, len(x_train) - 1)
      #print 'before load feature'
      features, labels = load_features(x_train[index:index + 16], y_train[index:index + 16])
      #print 'after load feature'
      #print(features)
      #print(labels)
      t = time.time()
      #print 'before run'
      _, loss_r = sess.run([train_op, loss],feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
      #print 'after run'
      elapsed = time.time() - t

      if step%10 == 0 and step != 0:
        
        print 'before summary'
	#print sess.run(debug_grad, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
        summary_str = sess.run(summary_op, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
	np_probs = sess.run(loss_vector, feed_dict={x:features, y:labels, keep_prob:FLAGS.keep_prob})
	print np_probs
        print labels
	summary_writer.add_summary(summary_str, step) 
        #print("time per batch is " + str(elapsed))
        print(step)
        print("loss for this image was " + str(loss_r))
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%200 == 9:
        total = 0
        false_pos = 0
        false_neg = 0
        true_pos = 0
        true_neg = 0
        avg_loss = 0.0
        total_accuracy = 0
        for i in range(len(x_dev)/16):
            begin = i * 16
            if random.uniform(0, 1) > SAMPLE_RATE:
                continue
            total += 1
            dev_features, dev_labels = load_features(x_dev[begin:begin + 16], y_dev[begin:begin + 16])
            preds = sess.run(correct_pred, feed_dict={x:dev_features, y:dev_labels, keep_prob:FLAGS.keep_prob})
            loss_r = sess.run(loss,feed_dict={x:dev_features, y:dev_labels, keep_prob:FLAGS.keep_prob})
            avg_loss += loss_r
            #print accuracy, y_dev[i]
            for enum_index, bool_ in enumerate(preds):
                if bool_:
                    if y_dev[begin + enum_index][1]:
                        true_pos += 1
                    else:
                        true_neg += 1
                else:
                    if y_dev[begin + enum_index][1]:
                        false_neg += 1
                    else:
                        false_pos += 1
            total_accuracy += float(sum(preds)) / len(preds)
        avg_loss /= len(x_dev)
        print ('total datapoints: ' + str(total))
        print ('true_pos: ' + str(true_pos))
        print ('true_neg: ' + str(true_neg))
        print ('false_pos: ' + str(false_pos))
        print ('false_neg: ' + str(false_neg))
        #print("accuracy: " + str(float(true_pos + true_neg) / total))
        print("accuracy: " + str(total_accuracy/total))
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
    total_accuracy = 0
    for i in range(len(x_test)/16):
        #if random.uniform(0, 1) > SAMPLE_RATE:
        #    continue
        begin = i * 16
        total += 1
        test_features, test_labels = load_features(x_test[begin:begin + 16], y_test[begin:begin + 16])
        preds = sess.run(correct_pred, feed_dict={x:test_features, y:test_labels, keep_prob:FLAGS.keep_prob})
        loss_r = sess.run(loss,feed_dict={x:test_features, y:test_labels, keep_prob:FLAGS.keep_prob})
        avg_loss += loss_r
        for enum_index, bool_ in enumerate(preds):
            if bool_:
                if y_test[begin + enum_index][1]:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if y_test[begin + enum_index][1]:
                    false_neg += 1
                else:
                    false_pos += 1
        total_accuracy += float(sum(preds)) / len(preds)
    avg_loss /= len(x_test)
    print ('total: ' + str(total))
    print ('true_pos: ' + str(true_pos))
    print ('true_neg: ' + str(true_neg))
    print ('false_pos: ' + str(false_pos))
    print ('false_neg: ' + str(false_neg))
    #print("accuracy: " + str(float(true_pos + true_neg) / total))
    print('accuracy: ' + str(total_accuracy/total))
    print('avg_loss: ' + str(avg_loss))
    print


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


