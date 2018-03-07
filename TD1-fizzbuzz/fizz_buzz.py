# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/

import numpy as np
import tensorflow as tf

NUM_DIGITS = 10

# codage binaire d'un chiffre (max NUM_DIGITS bits)
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# creation verite terrain: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

def fizz_buzz_index(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0


# donnees d'entrainement (X) et labels (Y)
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])


# definition du MLP a 1 couche cachee (non linearite ReLU)
# la fonction de cout (sortie finale) est definie separement
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Variables d'entree et de sortie du reseau
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# initialisation aleatoire des parametres (gaussienne)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

# fonction de prediction (estimation de la sortie du reseau)
py_x = model(X, w_h, w_o)

# Definition de l'apprentissage:
#   - fonction de cout (cross entropie sur softmax)
#   - methode de minimisation (descente de gradient)
# WARNING en python 3, il faut préciser (logits=py_x,labels=Y)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# prediction = plus grande (proba de) sortie
predict_op = tf.argmax(py_x, 1)

# affichage attendu par l'application
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def ground_true_fizz_buzz(i):
    if i%15==0:
        return "fizzbuzz"
    elif i%3==0:
        return "fizz"
    elif i%5==0:
        return "buzz"
    else:
        return str(i)

# on lance les calculs dans une "session"
BATCH_SIZE = 128 # taille minibatch
VALIDATION_SIZE=20 # Taille des données de validation
with tf.Session(  ) as sess:
    # tf.initialize_all_variables().run() # deprecated !
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        # melange des donnees a chaque 'epoch' (~iteration d'apprentissage)
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]


        # Apprentissage avec des minibatches de taille 128
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # affichage de la performance courante (1-erreur empirique)
        print(epoch, np.mean(np.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # Affichage sur les donnees de test
    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print(output)
    print(f"{np.average(teY==np.vectorize(fizz_buzz_index)(numbers)):0.0%} erreurs sur les données de test")
