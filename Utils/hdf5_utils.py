import h5py


def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        if isinstance(g[key], h5py.Dataset):
            yield f'{prefix}/{key}', key
        elif isinstance(g[key], h5py.Group):
            yield from h5py_dataset_iterator(g[key], f'{prefix}/{key}')
