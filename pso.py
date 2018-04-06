import numpy as np
import random
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Particle:
    def __init__(self, NUM_INPUTS):
        self.mData = np.random.uniform(low=-1, high=1, size=(NUM_INPUTS))
        self.mpBest = None
        self.mVelocity = np.zeros((NUM_INPUTS,))
        self.mBestError = 1e6

    def get_data(self):
        return self.mData

    def set_data(self, value):
        self.mData = value

    def get_pBest(self):
        return self.mpBest

    def set_pBest(self, value):
        self.mpBest = value

    def get_velocity(self):
        return self.mVelocity

    def set_velocity(self, velocity):
        self.mVelocity = velocity

    def tanh(self, x):
        e_plus = np.exp(x)
        e_minus = np.exp(-x)
        return (e_plus - e_minus) / (e_plus + e_minus)

    def set_best_error(self, value):
        self.mBestError = value

    def get_best_error(self):
        return self.mBestError

    def predict(self, X):
        W, b = self.get_weights_bias()
        Z = np.dot(W, X) + b
        A = self.tanh(Z)
        return A

    def get_weights_bias(self):
        W, b = self.mData[:-1], self.mData[-1]
        W = np.reshape(W, (1, -1))
        b = np.reshape(b, (1, 1))

        return W, b

    def get_mae(self, x_train, y_train):
        predicts = self.predict(x_train)

        return mean_absolute_error(predicts, y_train)

    def compute_error(self, X, y, X_valid, y_valid):
        W, b = self.mData[:-1], self.mData[-1]
        W = np.reshape(W, (1, -1))
        b = np.reshape(b, (1, 1))

        Z = np.dot(W, X) + b
        A = self.tanh(Z)
        error_train = 1. / X.shape[1] * np.sum(np.square(A - y))

        Z = np.dot(W, X_valid) + b
        A = self.tanh(Z)
        error_valid = 1. / X_valid.shape[1] * np.sum(np.square(A - y_valid))

        return 0.6 * error_valid + 0.4 * error_train


class PSO:
    def __init__(self, X, y, X_valid, y_valid, n_particles=100):
        self.n_inputs = X.shape[0]
        self.n_outputs = y.shape[0]
        self.X = X
        self.y = y
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.n_particles = n_particles
        self.c1 = 2
        self.c2 = 2
        self.v_max = 2
        self.v_min = -2
        self.w_max = 0.9
        self.w_min = 0.4

    def initialize_particles(self):
        length = self.n_inputs * self.n_outputs + self.n_outputs
        particles = []

        for i in range(self.n_particles):
            p = Particle(length)
            particles.append(p)

        return particles

    def train(self, epochs=100):
        particles = self.initialize_particles()
        gbest_error = 1e6
        gbest = None
        best_global_solutions = []

        for e in range(epochs):

            # w = (self.w_max - self.w_min) * (epochs - e) / epochs + self.w_min

            total_MAE = 0

            for p in particles:
                error = p.compute_error(self.X, self.y, self.X_valid, self.y_valid)

                total_MAE += 0.4 * p.get_mae(self.X, self.y) +  0.6 * p.get_mae(self.X_valid, self.y_valid)

                if error < p.get_best_error():
                    x = p.get_data()
                    copy_x = np.copy(x)
                    p.set_pBest(copy_x)
                    p.set_best_error(error)

                if gbest is None:
                    gbest = np.copy(p.get_data())
                    gbest_error = error

                    best_global_solutions.append(copy.deepcopy(p))

                else:
                    if error < gbest_error:
                        gbest = np.copy(p.get_data())
                        gbest_error = error

                        best_global_solutions.append(copy.deepcopy(p))

            print("AVG MAE: %.5f" % (total_MAE / self.n_particles))

            for p in particles:
                v_t = p.get_velocity()
                p_t = p.get_pBest()
                x_t = p.get_data()

                v_t = v_t + self.c1 * random.random() * (p_t - x_t) + self.c2 * random.random() * (gbest - x_t)

                v_t[v_t > self.v_max] = self.v_max
                v_t[v_t < self.v_min] = self.v_min

                x_t += v_t

        return best_global_solutions

