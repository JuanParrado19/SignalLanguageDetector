import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', value=0.):
    lengths = [len(seq) for seq in sequences]
    maxlen = maxlen or max(lengths)

    padded_sequences = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue  # Skip empty sequences
        if padding == 'post':
            padded_sequences[i, :len(seq)] = seq
        elif padding == 'pre':
            padded_sequences[i, -len(seq):] = seq
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return padded_sequences

data = pad_sequences(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
