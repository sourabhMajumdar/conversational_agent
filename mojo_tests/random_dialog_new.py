from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize
from sklearn import metrics
from memn2n import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf
import numpy as np
import os

tf.flags.DEFINE_float("learning_rate", 0.001,
                      "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10,
                        "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20,
                        "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks/",
                       "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/",
                       "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean('train', True, 'if True, begin to train')
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)


class chatBot(object):

    def __init__(self, data_dir, model_dir, task_id, isInteractive, OOV, memory_size, random_state, batch_size, learning_rate, epsilon, max_grad_norm, evaluation_interval, hops, epochs, embedding_size,session,description):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        # self.isTrain=isTrain
        self.isInteractive = isInteractive
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.sess = session

        self.description = description

        candidates, self.candid2indx = load_candidates(
            self.data_dir, self.task_id)
        self.n_cand = len(candidates)
        
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict(
            (self.candid2indx[key], key) for key in self.candid2indx)
        
        # task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, self.OOV)
        
        data = self.trainData + self.testData + self.valData
        
        self.build_vocab(data, candidates)
        # self.candidates_vec=vectorize_candidates_sparse(candidates,self.word_idx)
        
        self.candidates_vec = vectorize_candidates(
            candidates, self.word_idx, self.candidate_sentence_size)
        
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        
        #self.sess = tf.Session()

        num = input(" Printing all values for the MemN2N ")
        print(" batch_size : {}\n vocab_size : {}\n n_cand : {}\n sentence_size : {}\n embedding_size : {}\n candidates_vec : {}\n hops : {} max_grad_norm : {}\n task_id : {}\n".format(self.batch_size, self.vocab_size, self.n_cand, self.sentence_size,self.embedding_size,self.candidates_vec,self.hops,self.max_grad_norm,task_id))
        self.model = MemN2NDialog(self.batch_size, self.vocab_size, self.n_cand, self.sentence_size, self.embedding_size, self.candidates_vec, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=self.optimizer, task_id=self.task_id)
        self.saver = tf.train.Saver(max_to_keep=50)

        self.summary_writer = tf.summary.FileWriter(
            self.model.root_dir, self.model.graph_output.graph)


    def build_vocab(self, data, candidates):
        vocab = reduce(lambda x, y: x | y, (set(
            list(chain.from_iterable(s)) + q) for s, q, a in data))
        vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                             for candidate in candidates))
        vocab = sorted(vocab)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
        self.sentence_size = max(
            map(len, chain.from_iterable(s for s, _, _ in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        query_size = max(map(len, (q for _, q, _ in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(
            query_size, self.sentence_size)  # for the position
        # params
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length",
              self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)

    def interactive(self):
        context = []
        u = None
        r = None
        nid = 1
        while True:
            line = input('--> ').strip().lower()
            if line == 'exit':
                break
            if line == 'restart':
                context = []
                nid = 1
                print("clear memory")
                continue
            u = tokenize(line)
            data = [(context, u, -1)]
            s, q, a = vectorize_data(
                data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            preds = self.model.predict(s, q)
            r = self.indx2candid[preds[0]]
            print(r)
            r = tokenize(r)
            u.append('$u')
            u.append('#' + str(nid))
            r.append('$r')
            r.append('#' + str(nid))
            context.append(u)
            context.append(r)
            nid += 1

    def train(self):
        trainS, trainQ, trainA = vectorize_data(
            self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        valS, valQ, valA = vectorize_data(
            self.valData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        for t in range(1, self.epochs + 1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t = self.model.batch_fit(s, q, a)
                total_cost += cost_t
            if t % self.evaluation_interval == 0:
                train_preds = self.batch_predict(trainS, trainQ, n_train)
                val_preds = self.batch_predict(valS, valQ, n_val)
                train_acc = metrics.accuracy_score(
                    np.array(train_preds), trainA)
                val_acc = metrics.accuracy_score(val_preds, valA)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

                # write summary
                train_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'train_acc', tf.constant((train_acc), dtype=tf.float32))
                val_acc_summary = tf.summary.scalar(
                    'task_' + str(self.task_id) + '/' + 'val_acc', tf.constant((val_acc), dtype=tf.float32))
                merged_summary = tf.summary.merge(
                    [train_acc_summary, val_acc_summary])
                summary_str = self.sess.run(merged_summary)
                self.summary_writer.add_summary(summary_str, t)
                self.summary_writer.flush()

                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    self.saver.save(self.sess, self.model_dir +
                                    'model.ckpt', global_step=t)

    def converse(self) :

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        
        context = []
        u = None
        r = None
        nid = 1
        while True:
            line = input('--> ').strip().lower()
            if line == 'exit':
                break
            if line == 'restart':
                context = []
                nid = 1
                print("clear memory")
                continue
            u = tokenize(line)
            data = [(context, u, -1)]
            s, q, a = vectorize_data(
                data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            preds = self.model.predict(s, q)
            r = self.indx2candid[preds[0]]
            print(r)
            r = tokenize(r)
            u.append('$u')
            u.append('#' + str(nid))
            r.append('$r')
            r.append('#' + str(nid))
            context.append(u)
            context.append(r)
            nid += 1

    def converse_2(self,context,user_utterance,nid) :

        print(" hi from {} memory network ".format(self.description))

        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")

        u = None
        r = None

        line = user_utterance.strip().lower()
        
        u = tokenize(line)
        data = [(context, u, -1)]
        s, q, a = vectorize_data(
            data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        preds = self.model.predict(s, q)
        r = self.indx2candid[preds[0]]
        bot_response = r
        print(r)
        r = tokenize(r)
        u.append('$u')
        u.append('#' + str(nid))
        r.append('$r')
        r.append('#' + str(nid))
        context.append(u)
        context.append(r)
        nid += 1
        return context, nid


    def test(self):
        
        print(" Hey I am in the testing phase, model_dir is : ",self.model_dir)
        berger = input()
        
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testS, testQ, testA = vectorize_data(
                self.testData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            n_test = len(testS)
            print("Testing Size", n_test)
            test_preds = self.batch_predict(testS, testQ, n_test)
            test_acc = metrics.accuracy_score(test_preds, testA)
            print("Testing Accuracy:", test_acc)

    def batch_predict(self, S, Q, n):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            pred = self.model.predict(s, q)
            preds += list(pred)
        return preds

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    model_dir = "task" + str(FLAGS.task_id) + "_" + FLAGS.model_dir
    
    if not os.path.exists("main_mem_network/"):
        os.makedirs("main_mem_network")

    if not os.path.exists("loan_mem_network/"):
        os.makedirs("loan_mem_network")

    if not os.path.exists("balance_mem_network/"):
        os.makedirs("balance_mem_network")

    if not os.path.exists("transaction_mem_network/"):
        os.makedirs("transaction_mem_network")


    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    g4 = tf.Graph()

    main_session = tf.Session(graph=g1)
    loan_session = tf.Session(graph=g2)
    balance_session = tf.Session(graph=g3)
    transaction_session = tf.Session(graph=g4)

    print(" embedding_size for main_mem_network ",FLAGS.embedding_size)

    with g1.as_default() :
        main_mem_network = chatBot(data_dir=FLAGS.data_dir, 
                    model_dir="main_mem_network/",    
                    task_id=1,  
                    isInteractive=FLAGS.interactive,    
                    OOV=FLAGS.OOV,
                    memory_size=FLAGS.memory_size,
                    random_state=FLAGS.random_state,
                    batch_size=FLAGS.batch_size,
                    learning_rate=FLAGS.learning_rate,
                    epsilon=FLAGS.epsilon,
                    max_grad_norm=FLAGS.max_grad_norm,
                    evaluation_interval=FLAGS.evaluation_interval,
                    hops=FLAGS.hops,
                    epochs=FLAGS.epochs,
                    embedding_size=FLAGS.embedding_size,
                    session=main_session,
                    description="main")

    print(" embedding_size for loan_mem_network ",FLAGS.embedding_size)
    main = input("main_mem_network created successfully !!")

    with g2.as_default() :
        loan_mem_network = chatBot(data_dir=FLAGS.data_dir, 
                    model_dir="loan_mem_network/",    
                    task_id=1,  
                    isInteractive=FLAGS.interactive,    
                    OOV=FLAGS.OOV,
                    memory_size=FLAGS.memory_size,
                    random_state=FLAGS.random_state,
                    batch_size=FLAGS.batch_size,
                    learning_rate=FLAGS.learning_rate,
                    epsilon=FLAGS.epsilon,
                    max_grad_norm=FLAGS.max_grad_norm,
                    evaluation_interval=FLAGS.evaluation_interval,
                    hops=FLAGS.hops,
                    epochs=FLAGS.epochs,
                    embedding_size=FLAGS.embedding_size,
                    session=loan_session,
                    description="loan")

    loan = input("loan_mem_network created successfully !!")
    
    with g3.as_default() :
        balance_mem_network = chatBot(data_dir=FLAGS.data_dir, 
                    model_dir="balance_mem_network/",    
                    task_id=1,  
                    isInteractive=FLAGS.interactive,    
                    OOV=FLAGS.OOV,
                    memory_size=FLAGS.memory_size,
                    random_state=FLAGS.random_state,
                    batch_size=FLAGS.batch_size,
                    learning_rate=FLAGS.learning_rate,
                    epsilon=FLAGS.epsilon,
                    max_grad_norm=FLAGS.max_grad_norm,
                    evaluation_interval=FLAGS.evaluation_interval,
                    hops=FLAGS.hops,
                    epochs=FLAGS.epochs,
                    embedding_size=FLAGS.embedding_size,
                    session=balance_session,
                    description="balance")
    
    balance = input("balance_mem_network created successfully !!")
    
    with g4.as_default() :
        transaction_mem_network = chatBot(data_dir=FLAGS.data_dir, 
                    model_dir="transaction_mem_network/",    
                    task_id=FLAGS.task_id,  
                    isInteractive=FLAGS.interactive,    
                    OOV=FLAGS.OOV,
                    memory_size=FLAGS.memory_size,
                    random_state=FLAGS.random_state,
                    batch_size=FLAGS.batch_size,
                    learning_rate=FLAGS.learning_rate,
                    epsilon=FLAGS.epsilon,
                    max_grad_norm=FLAGS.max_grad_norm,
                    evaluation_interval=FLAGS.evaluation_interval,
                    hops=FLAGS.hops,
                    epochs=FLAGS.epochs,
                    embedding_size=FLAGS.embedding_size,
                    session=transaction_session,
                    description="transaction")

    transaction = input("transaction_mem_network created successfully !!")

    # chatbot.run()
    if FLAGS.train:
        with g1.as_default() :
            main_mem_network.train()
        with g2.as_default() :
            loan_mem_network.train()
        with g3.as_default():
            balance_mem_network.train()
        with g4.as_default() :
            transaction_mem_network.train()
    else:
        main_mem_network.test()
        loan_mem_network.test()
        balance_mem_network.test()
        transaction_mem_network.test()
    #chatbot.converse()

    network_dict = {"main" : [main_mem_network.converse_2,g1] , "loan" : [loan_mem_network.converse_2,g2], "balance" : [balance_mem_network.converse_2,g3], "transaction" : [transaction_mem_network.converse_2,g4]}

    story = list()
    nid = 1
    network_converse = main_mem_network.converse_2
    network_graph = g1
    while True :
        with network_graph.as_default() :
            user_utterance = input(" >>> : ")
            story, nid = network_converse(story,user_utterance,nid)
            if nid%3 == 0 :
                network_converse = network_dict["loan"][0]
                network_graph = network_dict["loan"][1]
            if nid%5 == 0 :
                network_converse = network_dict["balance"][0]
                network_graph = network_dict["balance"][1]
            if nid%7 == 0 :
                network_converse = network_dict["transaction"][0]
                network_graph = network_dict["transaction"][1]
            if nid%9 == 0 :
                network_converse = network_dict["main"][0]
                network_graph = network_dict["main"][1]

    #chatbot.close_session()
    main_mem_network.close_session()
    loan_mem_network.close_session()
    balance_mem_network.close_session()
    transaction_mem_network.close_session()
