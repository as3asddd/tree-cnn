from __future__ import division
from utils import *
import collections
import numpy as np
import tensorflow as tf
from root_node import root_network
from branch_node import branch_network
best_root_name = None
best_branch1_name = None
best_branch2_name = None
def initial_learn(root_net,branch_net1,branch_net2):
    global best_root_name, best_branch1_name, best_branch2_name
    _, best_root_name = root_net.train()
    _, best_branch1_name = branch_net1.train()
    _, best_branch2_name = branch_net2.train()
    print (best_root_name, best_branch1_name, best_branch2_name)
    return root_net,branch_net1,branch_net2

def incremental_learn(root,branch1,branch2,g1,g2,g3,new_class):
    global best_root_name, best_branch1_name, best_branch2_name
    dataset=data_process()
    idx=np.arange(0,5000)
    np.random.shuffle(idx)
    new_class_data=[]
    airplane=dataset.train_images[np.where(dataset.train_labels==0)[0]][:500]
    new_class_data.append(airplane)
    np.random.shuffle(idx)
    bird=dataset.train_images[np.where(dataset.train_labels==2)[0]][:500]
    new_class_data.append(bird)
    np.random.shuffle(idx)
    deer=dataset.train_images[np.where(dataset.train_labels==4)[0]][:500]
    new_class_data.append(deer)
    np.random.shuffle(idx)
    frog=dataset.train_images[np.where(dataset.train_labels==6)[0]][:500]
    new_class_data.append(frog)
    new_class_label=[0,2,4,6]
    branch1_class=[3,5,7]
    branch2_class=[1,8,9]
    with g1.as_default():
        saver = tf.train.import_meta_graph(best_root_name + '.meta')
        graph = tf.get_default_graph()
        image = graph.get_tensor_by_name('Placeholder:0')
        fc3 = graph.get_tensor_by_name('dense_2/BiasAdd:0')
        logits = tf.argmax(tf.nn.softmax(fc3), 1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options) ,graph=g1) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, best_root_name)
        while(1):
            if new_class==0:
                break
            predict=sess.run(logits,
							 feed_dict={image:new_class_data[len(new_class_label)-new_class],})
            predict_num=collections.Counter(predict)
            predict_probility={list(predict_num.keys())[i]:list(predict_num.values())[i]/500.0 for i in range(len(predict_num))}
            index=0
            branch_node=[]
            for i in range(len(predict_probility)):
                if list(predict_probility.values())[i]>0.5:
                    index+=1
                    branch_node.append(i)
            if index==0:
                root.root_class+=1
            if index==1:
                if branch_node[0]==0:
                    branch1.num_class+=1
                    branch1_class.append(new_class_label[len(new_class_label)-new_class])
                else:
                    branch2.num_class+=1
                    branch2_class.append(new_class_label[len(new_class_label)-new_class])
            if index>1:
                pass
            new_class-=1
    print(branch1_class, branch1.num_class)
    print(branch2_class, branch2.num_class)
    branch1_best_test_acc = fine_tune('branch1_initial_variables', branch1.num_class,branch1_class,g2, best_branch1_name)
    branch2_best_test_acc = fine_tune('branch2_initial_variables', branch2.num_class,branch2_class,g3, best_branch2_name)
    print ("branch 1 best test acc : {}".format(branch1_best_test_acc))
    print ("branch 2 best test acc : {}".format(branch2_best_test_acc))
def fine_tune(graph_name,num_class,branch_class,g, model_name):
    with g.as_default():
        saver=tf.train.import_meta_graph(model_name + '.meta')
        graph=tf.get_default_graph()
        image=graph.get_tensor_by_name('Placeholder:0')
        if graph_name=='branch1_initial_variables':
            keep_prob=graph.get_tensor_by_name('PlaceholderWithDefault_2:0')
            train_mode=graph.get_tensor_by_name('PlaceholderWithDefault_3:0')
            fc1=graph.get_tensor_by_name('branch1/dropout_3/Identity:0')
        else:
            keep_prob=graph.get_tensor_by_name('PlaceholderWithDefault_4:0')
            train_mode=graph.get_tensor_by_name('PlaceholderWithDefault_5:0')
            fc1 = graph.get_tensor_by_name('branch2/dropout_3/Identity:0')
        target=tf.placeholder(tf.int64,[None])
        
        fc3=tf.layers.dense(fc1,num_class)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,logits=fc3))

        optimizer=tf.train.AdamOptimizer(0.001, name='fine_tune').minimize(cross_entropy)
        correct=tf.equal(target,tf.argmax(tf.nn.softmax(fc3),axis=1))
        accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
        best_acc = 0
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_name)
            dataset=data_process()
            for epoch in range(20):
                x_batch,y_batch=dataset.fine_tune_next_batch(100,branch_class)
                print ("keep prob", keep_prob.eval(), "train_mode: ", train_mode.eval())
                for step in range(len(x_batch)):
                    feed_dict={image:x_batch[step],target:y_batch[step],keep_prob:[0.25,0.5],train_mode:True}
                    loss,_,acc=sess.run([cross_entropy,optimizer,accuracy],feed_dict=feed_dict)
                    if step%10==0:
                        print("number epoch %d,number step %d,cross entropy is %f"%(epoch,step,loss))
                        print("number epoch %d,number step %d,accuracy is %f"%(epoch,step,acc))
                test_accuracy=0
                x_batch,y_batch=dataset.fine_tune_next_batch(100,branch_class,mode='test')
                print (branch_class)
                for step in range(len(x_batch)):
                    feed_dict={image:x_batch[step],target:y_batch[step]}
                    acc=sess.run(accuracy,feed_dict=feed_dict)
                    test_accuracy+=acc
                test_accuracy = test_accuracy / len(x_batch)
                print("test accuracy is %f" % test_accuracy)
                if (test_accuracy > best_acc):
                    best_acc = test_accuracy
        return best_acc 

def main():
    root_initial_image=tf.placeholder(tf.float32,[None,32,32,3])
    root_initial_label=tf.placeholder(tf.int64,[None])
    branch_initial_label=tf.placeholder(tf.int64,[None])
    g1=tf.Graph()
    g2=tf.Graph()
    g3=tf.Graph()
    root_net=root_network(root_initial_image,root_initial_label)
    branch_net1=branch_network(root_initial_image,branch_initial_label,3,name='branch1')
    branch_net2=branch_network(root_initial_image,branch_initial_label,3,name='branch2')
    root_net,branch_net1,branch_net2=initial_learn(root_net,branch_net1,branch_net2)
    incremental_learn(root_net,branch_net1,branch_net2,g1,g2,g3,4)

if __name__=='__main__':
    main()
