import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys, random, memory, time 
from prepro import prepro
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


MEM_SIZE = 100000
BATCH_SIZE = 5
SGD1_STEPS = 5
SGD2_STEPS = 5
COLLECT_SIZE = 10 #num steps when collecting data
y = .99
e = 1.0
E_MULTIPLIER = 0.995
E_MIN = 0.05
num_episodes = 600
alpha1 = 0.0
alpha2 = 0.0
LRATE = 0.01

ACTION_REP = 4 # number of times the same action is repeated appart from the first time
MAX_CLIP = 10000.
MIN_CLIP = -10000.

UP = 2
DOWN = 3

# H network
# input: X placeholder
# output: Y placeholder, Yout network output
# parameters: H1.. hb1... HO
# loss: supervisedlossQ
# penalty: penaltyH = alpha1*regularizerH

# Q network
# input: X2 placeholder
# output: Z placeholder, Qout network output
# parameters: W1.. wb1... WO
# loss: supervisedlossQ
# penalty: penaltyQ=alpha2*regularizerQ



#Initializing things related to environment

env = gym.make('Pong-v0')
num_actions = 2



#Initializing things related to Qnetwork with tensorflow 

tf.reset_default_graph()

with tf.name_scope('inputs'):
    # inputs1 = tf.placeholder(shape=[1,4], dtype=tf.float32, name='input_state')
    # nextQ = tf.placeholder(shape=[1,2], dtype=tf.float32, name='nextQ')

    # inputs of network H
    X = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, num_actions], dtype=tf.float32, name='Y')

    # inputs of network Q
    X2 = tf.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32, name='X2')
    Z = tf.placeholder(shape=[None, num_actions], dtype=tf.float32, name='Z')

# Q network
with tf.name_scope('networkQ'):
    with tf.name_scope('layer1'):
        W1 = tf.get_variable('W1', shape=[8,8,4,32])
        wb1 = tf.Variable(tf.random_normal([32],0,0.2), name='wb1')
        conv_layer1 = tf.nn.relu(tf.nn.conv2d(X2, W1, strides=[1, 4, 4, 1], padding='SAME') + wb1)
        max_pool1 = tf.nn.max_pool(conv_layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        tf.histogram_summary(W1.name, W1)
        tf.histogram_summary(wb1.name,wb1)

    with tf.name_scope('layer2'):
        W2 = tf.Variable(tf.random_normal([4,4,32,64],0,0.2), name='W2')
        wb2 = tf.Variable(tf.random_normal([64],0,0.2), name='wb2')
        conv_layer2 = tf.nn.relu(tf.nn.conv2d(max_pool1, W2, strides=[1,2,2,1], padding='SAME') + wb2)
        max_pool2 = tf.nn.max_pool(conv_layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        tf.histogram_summary(W2.name, W2)
        tf.histogram_summary(wb2.name,wb2)

    with tf.name_scope('layer3'):
        W3 = tf.Variable(tf.random_normal([3,3,64,64],0,0.2), name='W3')
        wb3 = tf.Variable(tf.random_normal([64],0,0.2), name='wb3')
        conv_layer3 = tf.nn.relu(tf.nn.conv2d(max_pool2, W3, strides=[1,1,1,1], padding='SAME') + wb3)
        max_pool3 = tf.nn.max_pool(conv_layer3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        flat_layer3 = tf.reshape(max_pool3, [-1, 256])
        tf.histogram_summary(W3.name, W3)
        tf.histogram_summary(wb3.name,wb3)

    with tf.name_scope('outputlayer'):
        W4 = tf.Variable(tf.random_normal([256, 256], 0,0.2), name='W4')
        wb4 = tf.Variable(tf.random_normal([256],0,0.2), name='wb4')
        ff_layer4 = tf.nn.relu(tf.matmul(flat_layer3, W4) + wb4)

        W5 = tf.Variable(tf.random_normal([256, num_actions], 0,0.2), name='W5')
        Qout = tf.matmul(ff_layer4,W5, name='output_Qvalues')
        tf.histogram_summary(W5.name, W5)

# get best action
predict = tf.argmax(Qout,1,name='argmaxQout')

# Creating h network
with tf.name_scope('networkH'):
    with tf.name_scope('layer1'):
        H1 = tf.Variable(W1.initialized_value(),name='H1')
        hb1 = tf.Variable(wb1.initialized_value(),name='hb1')
        conv_layer1h = tf.nn.relu(tf.nn.conv2d(X, H1, strides=[1, 4, 4, 1], padding='SAME') + hb1)
        max_pool1h = tf.nn.max_pool(conv_layer1h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        tf.histogram_summary(H1.name, H1)
        tf.histogram_summary(hb1.name,hb1)
    with tf.name_scope('layer2'):
        H2 = tf.Variable(W2.initialized_value(),name='H2')
        hb2 = tf.Variable(wb2.initialized_value(),name='hb2')
        conv_layer2h = tf.nn.relu(tf.nn.conv2d(max_pool1h, H2, strides=[1,2,2,1], padding='SAME') + hb2)
        max_pool2h = tf.nn.max_pool(conv_layer2h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        tf.histogram_summary(H2.name, H2)
        tf.histogram_summary(hb2.name,hb2)
    with tf.name_scope('layer3'):
        H3 = tf.Variable(W3.initialized_value(),name='H3')
        hb3 = tf.Variable(wb3.initialized_value(),name='hb3')
        conv_layer3h = tf.nn.relu(tf.nn.conv2d(max_pool2h, H3, strides=[1,1,1,1], padding='SAME') + hb3)
        max_pool3h = tf.nn.max_pool(conv_layer3h, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        flat_layer3h = tf.reshape(max_pool3h, [-1, 256])
        tf.histogram_summary(H3.name, H3)
        tf.histogram_summary(hb3.name,hb3)
    with tf.name_scope('output_layer'):
        H4 = tf.Variable(W4.initialized_value(), name='H4')
        hb4 = tf.Variable(wb4.initialized_value(), name='hb4')
        ff_layer4h = tf.nn.relu(tf.matmul(flat_layer3h, H4) + hb4)
        H5 = tf.Variable(W5.initialized_value(), name='H5')
        Hout = tf.matmul(ff_layer4h,H5, name='output_Qvalues')
        tf.histogram_summary(H5.name, H5)
# @profile
def copy_H2Q():
    sess.run([W1.assign(H1), 
              wb1.assign(hb1), 
              W2.assign(H2),
              wb2.assign(hb2),
              W3.assign(H3), 
              wb3.assign(hb3), 
              WO.assign(HO)])

# Loss and Train
regularizerQ =  tf.reduce_sum(tf.square(W1)) + \
                tf.reduce_sum(tf.square(wb1)) + \
                tf.reduce_sum(tf.square(W2)) +  \
                tf.reduce_sum(tf.square(wb2)) +\
                tf.reduce_sum(tf.square(W3)) +\
                tf.reduce_sum(tf.square(wb3)) +\
                tf.reduce_sum(tf.square(W4)) + \
                tf.reduce_sum(tf.square(wb4)) + \
                tf.reduce_sum(tf.square(W5))

regularizerH =  tf.reduce_sum(tf.square(H1)) + \
                tf.reduce_sum(tf.square(hb1)) + \
                tf.reduce_sum(tf.square(H2)) +  \
                tf.reduce_sum(tf.square(hb2)) +\
                tf.reduce_sum(tf.square(H3)) +\
                tf.reduce_sum(tf.square(hb3)) +\
                tf.reduce_sum(tf.square(H4)) + \
                tf.reduce_sum(tf.square(hb4)) + \
                tf.reduce_sum(tf.square(H5))

with tf.name_scope('loss'):
    penaltyH = alpha1*regularizerH
    penaltyQ = alpha2*regularizerQ

    supervisedlossH = tf.reduce_mean(tf.square(Hout - Y)) + penaltyH
    supervisedlossQ = tf.reduce_mean(tf.square(Qout - Z)) + penaltyQ
    tf.scalar_summary('penaltyH', penaltyH)
    tf.scalar_summary('penaltyQ', penaltyQ)
    tf.scalar_summary('supervisedlossH', supervisedlossH)
    tf.scalar_summary('supervisedlossQ', supervisedlossQ)

with tf.name_scope('train'):
    ## defautl method for training without clipping
    # supervisedtrainH = tf.train.AdamOptimizer(learning_rate=0.001).minimize(supervisedlossH)
    # supervisedtrainQ = tf.train.AdamOptimizer(learning_rate=0.001).minimize(supervisedlossQ)

    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
    qvars = tvars[:9] # fetching only the variables in H network
    hvars = tvars[9:] # fetching only the variables in H network

    optimizerH = tf.train.AdamOptimizer(learning_rate=LRATE)
    gvsh = optimizerH.compute_gradients(supervisedlossH,hvars)
    capped_gvsh = [(tf.clip_by_value(grad, MIN_CLIP, MAX_CLIP ), var) for grad, var in gvsh]
    supervisedtrainH = optimizerH.apply_gradients(capped_gvsh)
    for grad, var in capped_gvsh:
        tf.histogram_summary('grad_' + var.name, grad)


    optimizerQ = tf.train.AdamOptimizer(learning_rate=LRATE)
    gvsq = optimizerQ.compute_gradients(supervisedlossQ,qvars)
    capped_gvsq = [(tf.clip_by_value(grad, MIN_CLIP, MAX_CLIP ), var) for grad, var in gvsq]
    supervisedtrainQ = optimizerQ.apply_gradients(capped_gvsq)
    for grad, var in capped_gvsq:
        tf.histogram_summary('grad_' + var.name, grad)






# @profile
def get_data(j,s,e,memory,Qout,COLLECT_SIZE,rAll):
    # j is current step
    # s is current state
    # e is current e-greedy parameter for random action
    # memory is the memory class
    # Qout is the NN that predicts the Qvalues
    # COLLECT_SIZE is the number of interactions we want to save in replay memory
    jmax = j+COLLECT_SIZE
    while (j < episode_length) and (j<jmax):
        j += 1
        a, allQ = sess.run([predict, Qout], feed_dict={X2:[s]})
        a[0] = UP if a[0] == 0 else DOWN
        rand_num = np.random.rand()
        if rand_num < e:
            a[0] = np.random.choice([UP, DOWN]) 

        # repeat same action ACTION_REP times   
        s1, r, d, _ = repeat_action(a, ACTION_REP)

        memory.add([s,a,r,s1,d]) # add interaction to memory buffer
        s = s1
        rAll += r
        if d == True:
            if e > E_MIN: e *= E_MULTIPLIER
            break
    return memory, j, s, d, rAll, e

# @profile
def create_batch_Hstep_targetTQ(memory,batch_size):
    '''
    Creates batches for the H network
    - the target of the H network is TQ
    - the input is called X2
    '''
    batch = memory.sample(batch_size)
    Xbatch = []
    Ybatch = []

    ## New code
    sb_list = []
    s1b_list = []
    db_list = []
    ab_list = []
    rb_list = []
    for index, transition in enumerate(batch):
        sb, ab, rb, s1b, db = transition 
        sb_list.append(sb)
        s1b_list.append(s1b)
        db_list.append(db)
        ab_list.append(ab)
        rb_list.append(rb)


    Q = sess.run(Qout, feed_dict={X2:sb_list})
    Q1 = sess.run(Qout, feed_dict={X2:s1b_list})
    maxQ1 = np.max(Q1, axis=1)
    targetQ = Q
    for i in range(len(batch)):
        action = ab_list[i]
        index_action = 0 if action[0] == UP else 1
        if db_list[i]:
            targetQ[i, index_action] = rb_list[i]
        else: 
            targetQ[i, index_action] = rb_list[i] + y*maxQ1[i]
        Xbatch.append(sb_list[i])
        Ybatch.append(targetQ[i])
    
    return Xbatch, Ybatch

# @profile
def create_batch_Qstep_targetH(memory,batch_size):
    # create batches for Q network
    # the target of the Q network is H* 
    batch = memory.sample(batch_size)
    Xbatch = []
    Ybatch = []

    sb_list = []
    for index, transition in enumerate(batch):
        sb, ab, rb, s1b, db = transition 
        sb_list.append(sb)

    targetH= sess.run(Hout, feed_dict={X:sb_list})

    for i in range(len(batch)):
        Xbatch.append(sb_list[i])
        Ybatch.append(targetH[i])

    return Xbatch, Ybatch

# @profile
def SGDH(Qout,Hout,memory,sgd_steps,batch_size):
    for i in range(sgd_steps):
        Xbatch, Ybatch = create_batch_Hstep_targetTQ(memory, batch_size)
        sess.run(supervisedtrainH, feed_dict={X:Xbatch, Y:Ybatch})

# @profile
def SGDQ(Qout,Hout,memory,sgd_steps,batch_size):
    for i in range(sgd_steps):
        X2batch, Zbatch = create_batch_Qstep_targetH(memory, batch_size)
        sess.run(supervisedtrainQ, feed_dict={X2:X2batch, Z:Zbatch})


# @profile
def repeat_action(a, ACTION_REP):
    r = 0
    k = 0
    s1 = []
    frame_d = False
    while k < ACTION_REP:
        k += 1
        if frame_d == False:
            frame_s1, frame_r, frame_d, _ = env.step(a[0])
            r += frame_r
        s1.append(prepro(frame_s1)) 

    s1 = np.rollaxis(np.array(s1), 0, 3 )
    return s1, r, frame_d, None

# @profile
def start_episode():
    s_raw = env.reset()
    s = []
    s.append(prepro(s_raw))  
    a = np.random.choice([UP, DOWN]) 

    for j in range(ACTION_REP-1):
        s1, r, d, _ = env.step(a)
        s.append(prepro(s1)) 
    s = np.rollaxis(np.array(s), 0, 3 )
    return s




mem = memory.memory(MEM_SIZE)

jList = []
rList = []

sess = tf.Session()
merged = tf.merge_all_summaries()

now = time.strftime("%c")
writer = tf.train.SummaryWriter('logs/'+now, sess.graph)

init = tf.initialize_all_variables()
sess.run(init)
total_step = 0
meanReward = 0

episode_length = 2000



for i in range(num_episodes):
    s = start_episode() 
    # env.render()
    rAll = 0
    d = False
    j = 0
    while j < episode_length:
        mem, j, s, d, rAll, e = get_data(j,s,e,mem,Qout,COLLECT_SIZE,rAll)
        SGDH(Qout,Hout, mem, sgd_steps=SGD1_STEPS, batch_size=BATCH_SIZE)
        SGDQ(Qout,Hout, mem, sgd_steps=SGD2_STEPS, batch_size=BATCH_SIZE)

        ## copy network
        # if step_count >= 2 or d:
        #     copy_H2Q()
        #     step_count = 0
        # step_count += 1

        if d==True:
            break

    
    print 'episode:', i, 'finished at j:', j 
    jList.append(j)
    rList.append(rAll)
    if len(rList)>10:
        meanReward = np.sum(rList[-10:])/10

    ## plot some statistics and games
    if i%5 == 0 :
        Xbatch, Ybatch = create_batch_Hstep_targetTQ(mem,10)
        X2batch, Zbatch = create_batch_Qstep_targetH(mem,10)
        result = sess.run(merged, feed_dict={X:Xbatch, Y:Ybatch, X2:X2batch, Z:Zbatch})
        writer.add_summary(result,i)
        print i, meanReward, 'e=',e
        for simulation in range(1):
            s = start_episode()
            d = False
            j = 0
            while not(d):
                j += 1 
                env.render()
                a = sess.run(predict, feed_dict={X2: [s]})
                a[0] = UP if a[0] == 0 else DOWN
                rand_num = np.random.rand()
                if rand_num < e:
                    a[0] = np.random.choice([UP, DOWN]) 

                s1, r, d, _ = repeat_action(a, ACTION_REP)
                
                s = s1

print "Average Reward per episode: " + str(sum(rList)/num_episodes)  
print "meanreward", meanReward

sess.close()


