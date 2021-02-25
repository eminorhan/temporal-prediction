import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Discretize cached embeddings')
parser.add_argument('--cache-path', default='', help='path to cached embeddings')
parser.add_argument('--nbins', default=64, type=int, help='number of bins along each embedding dimension (equal frequency bins)')

def equalbins(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))

if __name__ == '__main__':

    args = parser.parse_args()

    print(args)

    embeddings = np.load(args.cache_path)['x']
    print('Embeddings original shape:', embeddings.shape)

    ndim = embeddings.shape[1]

    # binning each embedding dimension separately
    for d in range(ndim): 
        eqbins = equalbins(embeddings[:, d], args.nbins)
        indices = np.digitize(embeddings[:, d], eqbins[1:], right=True)

        embeddings[:, d] = indices  # overwriting not to create another big embedding matrix

        # check to make sure the indices and frequencies are correct
        values, counts = np.unique(indices, return_counts=True)
        print('Values:', values)
        print('Counts:', counts)

        print('Completed dimension {:3d} of {:3d}'.format(d+1, ndim))
