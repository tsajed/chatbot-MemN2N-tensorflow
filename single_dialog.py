from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize
from sklearn import metrics
from memn2n import MemN2NDialog
from memn2n import MemN2NDialogHybrid
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
tf.flags.DEFINE_integer("memory_size", 100, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 6")
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
    def __init__(self, data_dir, model_dir, task_id, isInteractive=True, OOV=False, memory_size=50, random_state=None, batch_size=32, learning_rate=0.001, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3, epochs=200, embedding_size=20):
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
        self.set_max_sentence_length()
        # self.candidates_vec=vectorize_candidates_sparse(candidates,self.word_idx)

        self.trainS, self.trainQ, self.trainA = vectorize_data(
            self.trainData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        self.valS, self.valQ, self.valA = vectorize_data(
            self.valData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)

        self.candidates_vec = vectorize_candidates(
            candidates, self.word_idx, self.candidate_sentence_size)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()
        # Set max sentence vector size
        self.build_vocab(data, candidates)
        # Need to understand more about sentence size. Model failing because sentence size > candidate_sentence_size? Answers longer than queries?
        self.model = MemN2NDialog(self.batch_size, self.vocab_size, self.n_cand, self.max_sentence_size, self.embedding_size, self.candidates_vec, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer, task_id=task_id)
        # self.model = MemN2NDialogHybrid(self.batch_size, self.vocab_size, self.n_cand, self.max_sentence_size, self.embedding_size, self.candidates_vec, session=self.sess,
        #                           hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer, task_id=task_id)
        self.saver = tf.train.Saver(max_to_keep=50)

        self.summary_writer = tf.summary.FileWriter(
            self.model.root_dir, self.model.graph_output.graph)

    def set_max_sentence_length(self):
        if self.candidate_sentence_size > self.sentence_size:
            self.max_sentence_size = self.candidate_sentence_size
        else:
            self.max_sentence_size = self.sentence_size

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
            # Need to take care of the candidate sentence size > sentence size. In both main function and here
            # Whichever of candidate_size or candidate_sentence_size is higher, that should be allowed
            s, q, a = vectorize_data(
                data, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
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
        # trainS, trainQ, trainA = vectorize_data(
        #     self.trainData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        # valS, valQ, valA = vectorize_data(
        #     self.valData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_train = len(self.trainS)
        n_val = len(self.valS)

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
                s = self.trainS[start:end]
                q = self.trainQ[start:end]
                a = self.trainA[start:end]
                cost_t = self.model.batch_fit(s, q, a)
                total_cost += cost_t
            if t % self.evaluation_interval == 0:
                train_preds = self.batch_predict(self.trainS, self.trainQ, n_train)
                val_preds = self.batch_predict(self.valS, self.valQ, n_val)
                train_acc = metrics.accuracy_score(
                    np.array(train_preds), self.trainA)
                val_acc = metrics.accuracy_score(val_preds, self.valA)
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

    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testS, testQ, testA = vectorize_data(
                self.testData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
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
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(FLAGS.data_dir, model_dir, FLAGS.task_id, OOV=FLAGS.OOV,
                      isInteractive=FLAGS.interactive, batch_size=FLAGS.batch_size)
    # chatbot.run()
    if FLAGS.train:
        chatbot.train()
        print("OOV=" + str(FLAGS.OOV))
        chatbot.test()
    else:
        print("OOV=" + str(FLAGS.OOV))
        chatbot.test()
    chatbot.close_session()
