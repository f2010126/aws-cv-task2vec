import random
import string
import codecs
import numpy as np
from tqdm import tqdm

# document this function
def random_string(stringLength=5):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def load_embeddings():
    print('loading word embeddings...')
    embeddings_index = {}
    f = codecs.open('wiki.de.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    return embeddings_index