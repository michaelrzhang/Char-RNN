
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.utils.data_utils import get_file
import numpy as np
import random
import pickle
import sys


# Loosely follows Karparthy, Keras library example, and mineshmathew's repo
# I added code and comments for clarity and ease of use.


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array (from Keras library)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # Taking the log should be optional? add fudge factor to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def test_model(model, char_to_indices, indices_to_char, seed_string=" ", temperature=1.0, test_length=150):
    """
    Higher temperatures correspond to more potentially creative sentences (at the cost of mistakes)
    """
    num_chars = len(char_to_indices.keys())
    for i in range(test_length):
        test_in = np.zeros((1, len(seed_string), num_chars))
        for t, char in enumerate(seed_string):
            test_in[0, t, char_to_indices[char]] = 1
        # input 'goodby', desired output is 'oodbye' # possible todo: show that this holds for the model
        entire_prediction = model.predict(test_in, verbose=0)[0]
        next_index = sample(entire_prediction[-1], temperature)
        next_char = indices_to_char[next_index]
        seed_string = seed_string + next_char
    return seed_string


if __name__ == "__main__":

    # Parameters

    # Random seed. Change to get different training results / speeds
    # origin = "obama2"  # used to name files saved as well
    # origin = "nietzsche"
    origin = "sonnets"
    seed = 2

    # LSTM parameters (todo: GRU and RNN are other options)
    unit_size = 512  # can increase more if using dropout
    num_layers = 3
    dropout = 0.2

    # optimization parameters
    optimizer = 'rmsprop'
    training_epochs = 50

    # how we break sentences up
    maxlen = 120 # perhaps better for step not to divide maxlen (to get more overlap) 
    step = 13
    # increasing maxlen should allow for more coherent thoughts
    # previously maxlen = 40, step = 10 before

    # testing
    test_length = 150

    # Select source
    if "nietzsche" in origin:
        path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        text = open(path).read().lower()
    elif "obama" in origin:
        text = open("data/obama.txt").read().lower()
    elif "sonnet" in origin:
        text = open("data/sonnet.txt").read().lower() 
    elif "paulgraham" in origin:
        text = open("data/paulgraham.txt").read().lower()
    else:  # add your own text! (something to do with sports would be interesting)
        raise NotImplementedError

    np.random.seed(seed)
    random.seed(seed)

    print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    num_chars = len(chars)
    print('characters: ', chars)
    print('total characters in vocabulary:', num_chars)

    # dictionaries to convert characters to numbers and vice-versa
    char_to_indices = dict((c, i) for i, c in enumerate(chars))
    indices_to_char = dict((i, c) for i, c in enumerate(chars))
    pickle.dump(char_to_indices, open("saved_models/{}c2i.p".format(origin), "wb"))
    pickle.dump(indices_to_char, open("saved_models/{}i2c.p".format(origin), "wb"))

    # cut the text in semi-redundant sequences of maxlen characters (possible todo: try cuts of different sizes)
    sentences = []
    targets = []
    for i in range(0, len(text) - maxlen - 1, step):
        sentences.append(text[i: i + maxlen])
        targets.append(text[i + 1: i + maxlen + 1])
    print('number of sequences:', len(sentences))

    print('Vectorization...')
    """
    One reason to do this is that entering raw numbers into a RNN may not make sense
    because it assumes an ordering for catergorical variables
    """
    X = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
    y = np.zeros((len(sentences), maxlen, num_chars), dtype=np.bool)
    for i in range(len(sentences)):
        sentence = sentences[i]
        target = targets[i]
        for j in range(maxlen):
            X[i][j][char_to_indices[sentence[j]]] = 1
            y[i][j][char_to_indices[target[j]]] = 1

    print('Building model...')
    model = Sequential()

    # model.add(LSTM(unit_size, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(LSTM(unit_size, input_dim=num_chars, return_sequences=True))
    for i in range(num_layers - 1):
        if dropout:  # as proposed by Zaremba et al. (may want to have dropout before first LSTM cell as well?)
            model.add(Dropout(dropout))
        model.add(LSTM(unit_size, return_sequences=True))
    if dropout:
        model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(num_chars)))  # todo: shouldn't have to be same size right?
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  )
    print(model.summary())
    print('...model built!')

    # saves generated text in file
    outfile = open("generated/{}.txt".format(origin), "w")
    # -----Training-----
    for i in range(training_epochs):
        print('-' * 10 + ' Iteration: {} '.format(i) + '-' * 10)
        outfile.write("\n" + '-' * 10 + ' Iteration: {} '.format(i) + '-' * 10 + "\n")
        for temperature in [0.03, 0.1, 0.3, 1, 2]:
            generated_string = test_model(model,
                                          char_to_indices=char_to_indices,
                                          indices_to_char=indices_to_char,
                                          temperature=temperature,
                                          test_length=test_length)
            output = "Temperature: {} Generated string: {}".format(temperature, generated_string)
            print(output)
            outfile.write(output + "\n")
            outfile.flush()

        history = model.fit(X, y, batch_size=128, epochs=1, verbose=0)
        print('loss is {}'.format(history.history['loss'][0]))

    outfile.close()
    model.save("saved_models/{}.h5".format(origin))
