import argparse 
import tensorflow as tf
from load_graph import load_graph
import dataset_loading
import numpy as np
import functions

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="model/affordance-cvae-27000-2-final.ckpt.bytes", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # load data
    vr_data = dataset_loading.read_timeseries_data_only_train('VR_data')

    # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    input_dims = 27000
    cond_dims = 25
    X_in = graph.get_tensor_by_name('prefix/X:0')
    Cond = graph.get_tensor_by_name('prefix/Cond:0')
    dec = graph.get_tensor_by_name('prefix/decoder/Reshape_1:0')
    keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
    batch_size = 64
    # We launch a Session
    predictions = []
    batches = []
    labels = []
    errors = []
    batches_completed = 0
    print(vr_data.train.num_examples)
    with tf.Session(graph=graph) as sess:
        while vr_data.train.num_examples - batches_completed * batch_size > 0:
            batch, batch_labels = vr_data.train.next_timeseries_batch(batch_size=batch_size)
            batch = np.reshape(batch, [-1, input_dims])
            batch_labels = np.reshape(batch_labels, [-1, cond_dims])
            d = sess.run([dec], feed_dict={X_in: batch, Cond: batch_labels, keep_prob: 0.8})
            predictions.append(d[0])
            batches.append(batch)
            labels = labels + batch_labels.tolist()
            e = functions.root_mean_square_error(d[0], batch)
            errors = errors + e
            batches_completed += 1

    sess.close()

    np_pred = np.array(predictions)
    np_batches = np.array(batches)
    np_errors = np.array(errors)
    np_labels = np.array(labels)
    np.save("predictions.npy", np_pred)
    np.save("batches.npy", np_batches)
    np.save("errors.npy", np_errors)
    np.save("labels.npy", np_labels)