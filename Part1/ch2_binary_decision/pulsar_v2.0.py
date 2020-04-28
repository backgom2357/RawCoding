import numpy as np
import csv
import time

# Hyperparameters

RND_MEAN = 0
RND_STD = 0.0030
LEARNING_RATE = 0.001
np.random.seed(1234)

def randomize(): np.random.seed(time.time())

class Pulsar:

    def __init__(self, adjust_ratio=False):
        self.input_cnt = 8
        self.output_cnt = 1
        self.weight, self.bias = self.init_model()
        self.data = self.load_pulsar_dataset(adjust_ratio)
        self.shuffle_map = np.arange(self.data.shape[0])

    def pulsar_exec(self, epoch_count=10, mb_size=10, report=1):
        self.train_and_test(epoch_count, mb_size, report)

    def load_pulsar_dataset(self, adjust_ratio):
        pulsars, stars = [], []
        with open('../../DataSets/pulsar_stars.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            for row in csvreader:
                if row[8] == '1':
                    pulsars.append(row)
                else:
                    stars.append(row)

        star_cnt, pulsar_cnt = len(stars), len(pulsars)

        if adjust_ratio:
            data = np.zeros([2 * star_cnt, 9])
            data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
            for n in range(star_cnt):
                data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype='float32')
            return data
        else:
            data = np.zeros([star_cnt + pulsar_cnt, 9])
            data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
            data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')
            return data

    def init_model(self):
        weight = np.random.normal(RND_MEAN, RND_STD, [self.input_cnt, self.output_cnt])
        bias = np.zeros([self.output_cnt])
        return weight, bias

    def train_and_test(self, epoch_count, mb_size, report):
        step_count, test_begin_idx = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data(test_begin_idx)

        for epoch in range(epoch_count):
            losses = []

            for n in range(step_count):
                train_x, train_y = self.get_train_data(mb_size, n, test_begin_idx)
                loss, _ = self.run_train(train_x, train_y)
                losses.append(loss)

            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
                print('Epoch {}: loss={:5.3f}, result={}'.format(epoch + 1, np.mean(losses), acc_str))

        acc = self.run_test(test_x, test_y)
        acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
        print('\nFinal Test: final result = {}'.format(acc_str))

    def arrange_data(self, mb_size):
        np.random.shuffle(self.shuffle_map)
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        test_begin_idx = step_count * mb_size
        return step_count, test_begin_idx

    def get_test_data(self, test_begin_idx):
        test_data = self.data[self.shuffle_map[test_begin_idx:]]
        return test_data[:, :-self.output_cnt], test_data[:, -self.output_cnt:]


    def get_train_data(self, mb_size, nth, test_begin_idx):
        if nth == 0:
            np.random.shuffle(self.shuffle_map[:test_begin_idx])
        train_data = self.data[self.shuffle_map[mb_size*nth:mb_size*(nth+1)]]
        return train_data[:, :-self.output_cnt], train_data[:, -self.output_cnt:]


    def run_train(self, x, y):
        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy


    def forward_neuralnet(self, x):
        output = np.matmul(x, self.weight) + self.bias
        return output, x


    def forward_postproc(self, output, y):
        entropy = self.sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        return loss, [y, output, entropy]


    def backprop_postproc(self, G_loss, aux):
        y, output, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output


    def backprop_neuralnet(self, G_output, x):

        g_output_w = x.transpose()

        G_w = np.matmul(g_output_w, G_output)
        G_b = np.sum(G_output, axis=0)

        self.weight -= LEARNING_RATE * G_w
        self.bias -= LEARNING_RATE * G_b

    def eval_accuracy(self, output, y):
        est_yes = np.greater(output, 0)
        ans_yes = np.greater(y, 0.5)
        est_no = np.logical_not(est_yes)
        ans_no = np.logical_not(ans_yes)

        tp = np.sum(np.logical_and(est_yes, ans_yes))
        fp = np.sum(np.logical_and(est_yes, ans_no))
        fn = np.sum(np.logical_and(est_no, ans_no))
        tn = np.sum(np.logical_and(est_no, ans_yes))

        accuracy = self.safe_div(tp + fn, tp + tn + fp + fn)
        precision = self.safe_div(tp, tp + fp)
        recall = self.safe_div(tp, tp + tn)
        f1 = 2 * self.safe_div(recall * precision, recall + precision)

        return [accuracy, precision, recall, f1]

    def safe_div(self, p, q):
        p, q = float(p), float(q)
        if np.abs(q) < 1.0e-20: return np.sign(p)
        return p / q

    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        return np.exp(-self.relu(-x)) / (1.0 + np.exp(-np.abs(x)))

    def sigmoid_derv(self, x, y):
        return y * (1 - y)

    def sigmoid_cross_entropy_with_logits(self, z, x):
        return self.relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

    def sigmoid_cross_entropy_with_logits_derv(self, z, x):
        return -z + self.sigmoid(x)




if __name__=="__main__":
    pulsar = Pulsar(adjust_ratio=True)
    pulsar.pulsar_exec()



