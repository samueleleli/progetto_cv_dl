import pickle

with open('data/arch-hdf5-test/dict_max_values.pickle', 'rb') as handle:
    dict_max_values = pickle.load(handle)

print(dict_max_values.keys())