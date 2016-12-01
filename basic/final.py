import tensorflow as tf
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("variables"):
        # Variable that track how many time the graph has been run
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        # Variable that keeps track of sum of all output values over time:
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    with tf.name_scope("transformation"):
        # Create input placeholder - takes in a vector of any length
        a = tf.placeholder(tf.float32, shape=[None], name="input_place-holder_a")

        # Separate middle layer
        with tf.name_scope("intermidiate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="add_c")

        # Separate output layer
        with tf.name_scope("output"):
            output = tf.add(b,c, name="output")

    with tf.name_scope("update"):
        # Increments the total_output Variable by the lastest input
        update_total = total_output.assign_add(output)

        # Increments the `global_step` Variable, shoulbe run wheneber the graph is run
        increment_step = global_step.assign_add(1)
    
    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")

        # Create summaries for output node
        tf.scalar_summary(b'Output', output, name="output_summary")
        tf.scalar_summary(b'Sum of output over time', update_total, name="total_summary")
        tf.scalar_summary(b'Average of outputs over time', avg, name="average_summary")

    with tf.name_scope("global_ops"):
        # Initialization Op
        init = tf.initialize_all_variables()

        # Merge all summaries into one Operation
        merged_summaries = tf.merge_all_summaries()
sess = tf.Session(graph=graph)
writer = tf.train.SummaryWriter('./improved_graph', graph)

sess.run(init)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)

    writer.add_summary(summary, global_step=step)


run_graph([2,8])
run_graph([3,1,3,3])
run_graph([1,3,3])
run_graph([3])
run_graph([8])

writer.flush()
writer.close()
sess.close()

