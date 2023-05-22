###############################################################################
'''This parser takes as input the .h5 file produced by create_datafile.py and
outputs a .h5 file with datapoints of the form (X, Y), which can be understood
by Keras models.'''
###############################################################################

import sys
import time

import h5py
import numpy as np
from more_itertools import one

from constants import data_dir
from utils import create_datapoints

start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']
assert sys.argv[2] in ['0', '1', 'all']

h5f = h5py.File(data_dir + 'datafile'
                + '_' + sys.argv[1] + '_' + sys.argv[2]
                + '.h5', 'r')

SEQ = h5f['SEQ'][:]
STRAND = h5f['STRAND'][:]
TX_START = h5f['TX_START'][:]
TX_END = h5f['TX_END'][:]
JN_START = h5f['JN_START'][:]
JN_END = h5f['JN_END'][:]
h5f.close()

h5f2 = h5py.File(data_dir + 'dataset' + '_' + sys.argv[1] + '_' + sys.argv[2] + '.h5', 'w')

CHUNK_SIZE = 100

for i in range(SEQ.shape[0]//CHUNK_SIZE):
    # Each dataset has CHUNK_SIZE genes

    if (i+1) == SEQ.shape[0]//CHUNK_SIZE:
        NEW_CHUNK_SIZE = CHUNK_SIZE + SEQ.shape[0] % CHUNK_SIZE
    else:
        NEW_CHUNK_SIZE = CHUNK_SIZE

    X_batch = []
    Y_batch = [[] for t in range(1)]

    for j in range(NEW_CHUNK_SIZE):

        idx = i*CHUNK_SIZE + j

        seq = SEQ[idx].decode()
        strand = STRAND[idx].decode()
        jn_start = one(JN_START[idx]).decode()
        jn_end = one(JN_END[idx]).decode()
        X, Y = create_datapoints(
            seq, strand,
            TX_START[idx], TX_END[idx],
            jn_start, jn_end,
        )

        X_batch.extend(X)
        for t in range(1):
            Y_batch[t].extend(Y[t])

    X_batch = np.asarray(X_batch).astype('int8')
    for t in range(1):
        Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')

    h5f2.create_dataset('X' + str(i), data=X_batch)
    h5f2.create_dataset('Y' + str(i), data=Y_batch)

h5f2.close()

print(f"--- {time.time() - start_time} seconds ---")

###############################################################################
