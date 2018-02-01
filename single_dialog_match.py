from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data_match, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize, parse_kb
from sklearn import metrics
from memn2n import MemN2NDialog
from memn2n import MemN2NDialogMatch
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
tf.flags.DEFINE_boolean("match", False, "Use the match features [False]")
tf.flags.DEFINE_string("kb_file", "data/dialog-bAbI-tasks/dialog-babi-kb-all.txt", "KB file path")
tf.flags.DEFINE_float("random_time", 0.1, "Random time [0.1]")
FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)

def get_temporal_encoding(d, random_time=0.):
    te = []
    for i in range(len(d)):
        l = int(np.sign(d[i].sum(axis=1)).sum())
        try:
            data_shape = np.array(d).shape[1]
        except IndexError:
            data_shape = 14
        temporal_encoding = np.zeros(data_shape)
        if l != 0:
            if random_time > 0.:
                nblank = np.random.randint(0, np.ceil(l * random_time) + 1)
                rt = np.random.permutation(l + nblank) + 1 # +1: permutation starts from 0
                rt = np.vectorize(lambda x: data_shape if x > data_shape else x)(rt)
                temporal_encoding[:l] = np.sort(rt[:l])[::-1]
            else:
                temporal_encoding[:l] = np.arange(l, 0, -1)
        te.append(temporal_encoding)
    return te

kb_types = [
    'R_cuisine',
    'R_location',
    'R_price',
    'R_rating',
    'R_phone',
    'R_address',
    'R_number',
]

def get_kb_type_idx(t):
    assert t in kb_types
    return kb_types.index(t)

def get_kb_type(kb, word):
    for t, v in kb.items():
        if word in v:
            return t
    return None

def find_match_in_story(word, story):
    for s in story:
        if word in s:
            return True
    return False

def create_match_features(data, idx2ans, kb):
    ret = np.zeros((len(data), 7, len(idx2ans)))
    for i, (story, _, _) in enumerate(data):
        for j in range(len(idx2ans)):
            a = idx2ans[j].split(' ')
            m = np.zeros(7)
            for w in a:
                kb_type = get_kb_type(kb, w)
                if kb_type and find_match_in_story(w, story):
                    m[get_kb_type_idx(kb_type)] = 1
            ret[i, :, j] = m
    return ret

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
        self.trainS, self.trainQ, self.trainA = vectorize_data_match(
            self.trainData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        self.valS, self.valQ, self.valA = vectorize_data_match(
            self.valData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)

        self.candidates_vec = vectorize_candidates(
            candidates, self.word_idx, self.candidate_sentence_size)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()
        # Set max sentence vector size
        self.build_vocab(data, candidates)

        answer_n_hot = np.zeros((self.vocab_size, len(self.candid2indx)))
        for ans_it in range(len(self.indx2candid)):
            ans = self.indx2candid[ans_it]
            n_hot = np.zeros((self.vocab_size, ))
            for w in tokenize(ans):
                assert w in self.word_idx
                n_hot[self.word_idx[w]] = 1
            answer_n_hot[:, ans_it] = n_hot

        # Need to understand more about sentence size. Model failing because sentence size > candidate_sentence_size? Answers longer than queries?
        self.model = MemN2NDialogMatch(self.batch_size, self.vocab_size, self.max_sentence_size, self.memory_size, self.embedding_size, answer_n_hot, match=FLAGS.match, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer, task_id = self.task_id)
        # self.model = MemN2NDialogHybrid(self.batch_size, self.vocab_size, self.n_cand, self.max_sentence_size, self.embedding_size, self.candidates_vec, session=self.sess,
        #                           hops=self.hops, max_grad_norm=self.max_grad_norm, optimizer=optimizer, task_id=task_id)
        self.saver = tf.train.Saver(max_to_keep=50)

        self.summary_writer = tf.summary.FileWriter(
            self.model.root_dir, self.model.graph_output.graph)

        self.kb = parse_kb(FLAGS.kb_file)

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
            s, q, a = vectorize_data_match(
                data, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
            m = None
            if FLAGS.match:
                m = create_match_features([(s,q,a)], self.indx2candid, self.kb)
            preds = self.model.predict(s, q, get_temporal_encoding(s, random_time=0.0), False, m)

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
        # trainS, trainQ, trainA = vectorize_data_match(
        #     self.trainData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        # valS, valQ, valA = vectorize_data_match(
        #     self.valData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_train = len(self.trainS)
        n_val = len(self.valS)
        batch_size = self.batch_size

        trainM, valM, testM = None, None, None
        if FLAGS.match:
            #logger.info("Building match features for training set ...")
            trainM = create_match_features(self.trainData, self.indx2candid, self.kb)
            #logger.info("Done")
            #logger.info("Building match features for validation set ...")
            valM = create_match_features(self.valData, self.indx2candid, self.kb)
            #logger.info("Done")
            #logger.info("Building match features for test set ...")
            #testM = create_match_features(self.testData, self.indx2candid, self.kb)
            #logger.info("Done")

        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size),
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0

        train_labels = np.argmax(self.trainA, axis=1)
        #test_labels = np.argmax(testA, axis=1)
        val_labels = np.argmax(self.valA, axis=1)

        for t in range(1, self.epochs + 1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = self.trainS[start:end]
                q = self.trainQ[start:end]
                a = self.trainA[start:end]
                m = trainM[start:end] if FLAGS.match else None
                temporal = get_temporal_encoding(s, random_time=FLAGS.random_time)
                cost_t = self.model.batch_fit(s, q, a, temporal, False, m)
                total_cost += cost_t

            if t % self.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, self.batch_size):
                    end = start + self.batch_size
                    s = self.trainS[start:end]
                    q = self.trainQ[start:end]
                    m = trainM[start:end] if FLAGS.match else None
                    temporal = get_temporal_encoding(s, random_time=0.0)
                    pred = self.model.predict(s, q, temporal, False, m)
                    train_preds += list(pred)
    
                #val_preds = self.model.predict(valS, valQ, get_temporal_encoding(valS, random_time=0.0), True, valM)
                val_preds = self.batch_predict(self.valS, self.valQ, n_val, valM)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                val_acc = metrics.accuracy_score(val_preds, val_labels)
                
                last_train_acc = train_acc
                last_val_acc = val_acc

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
            testS, testQ, testA = vectorize_data_match(
                self.testData, self.word_idx, self.max_sentence_size, self.batch_size, self.n_cand, self.memory_size)
            testM = None
            if FLAGS.match:
                testM = create_match_features(self.testData, self.indx2candid, self.kb)
            n_test = len(testS)
            test_labels = np.argmax(testA, axis=1)
            print("Testing Size", n_test)
            test_preds = self.batch_predict(testS, testQ, n_test, testM)
            test_acc = metrics.accuracy_score(test_preds, test_labels)
            # test_preds = self.batch_predict(testS, testQ, n_test)
            # test_acc = metrics.accuracy_score(test_preds, testA)
            print("Testing Accuracy:", test_acc)

    def batch_predict(self, S, Q, n, M):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            m = M[start:end] if FLAGS.match else None
            pred = self.model.predict(s, q, get_temporal_encoding(s, random_time=0.0), False, m)
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
        print("match=" + str(FLAGS.match))
        chatbot.test()
    else:
        print("OOV=" + str(FLAGS.OOV))
        print("match=" + str(FLAGS.match))
        chatbot.test()
    chatbot.close_session()
