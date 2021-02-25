import argparse
import numpy as np
from sklearn.decomposition import IncrementalPCA

parser = argparse.ArgumentParser(description='Dim reduction for cached embeddings')
parser.add_argument('--model', default='say', choices=['say', 'in', 'wsl', 'rand', 'robust', 'spatial'], help='embedding model name')
parser.add_argument('--data', default='adept', choices=['adept', 'intphys'], help='cached data to be dim-reduced')
parser.add_argument('--ndim', default=256, type=int, help='reduced dimensionality')

args = parser.parse_args()

print(args)

if __name__ == '__main__':

    if args.data == 'adept':
        embeddings = np.load('../caches_adept/adept_train_' + args.model + '.npz')['x']
    elif args.data == 'intphys':
        embeddings = np.load('../caches_intphys/noise_0.1/intphys_train_' + args.model + '.npz')['x']

    print('Embeddings original shape:', embeddings.shape)

    streamed_embeddings = np.reshape(embeddings, (-1, 2048))
    print('Embeddings streamed shape:', streamed_embeddings.shape)

    # do incremental PCA
    ipca = IncrementalPCA(n_components=args.ndim)
    ipca.fit_transform(streamed_embeddings)
    print('PCA fitted')

    components = ipca.components_

    new_embeddings = np.dot(embeddings, components.T)
    print('Successfully completed the online PCA. Variance explained:', np.sum(ipca.explained_variance_ratio_))
    print('New embeddings shape:', new_embeddings.shape)
    print('Components shape:', components.shape)

    savefile_name = args.data + '_train_' + args.model + '_' + str(args.ndim)

    np.savez(savefile_name, x=new_embeddings, W=components)
