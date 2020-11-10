import os, sys

import torch
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DelfPCA():
    def __init__(self, pca_n_components,
                 whitening=True,
                 pca_saved_path=None):
        self.pca_n_components = pca_n_components
        self.whitening=True
        self.pca_saved_path = os.path.join(pca_saved_path, 'pca.h5')
    
    def __call__(self, features):
        pca = PCA(whiten=self.whitening)
        pca.fit(np.array(features))
        pca_matrix = pca.components_
        pca_mean = pca.mean_
        pca_var = pca.explained_variance_

        # save as h5 file.
        print('================= PCA RESULT ==================')
        print('pca_matrix: {}'.format(pca_matrix.shape))
        print('pca_mean: {}'.format(pca_mean.shape))
        print('pca_vars: {}'.format(pca_var.shape))
        print('===============================================')

        # save features, labels to h5 file.
        filename = os.path.join(self.pca_saved_path)
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('pca_matrix', data=pca_matrix)
        h5file.create_dataset('pca_mean', data=pca_mean)
        h5file.create_dataset('pca_vars', data=pca_var)
        h5file.close()
         

def ApplyPcaAndWhitening(data, pca_matrix, pca_mean, pca_vars, pca_dims, use_whitening=False):
    '''apply PCA/Whitening to data.
    Args: 
        data: [N, dim] FloatTensor containing data which undergoes PCA/Whitening.
        pca_matrix: [dim, dim] numpy array PCA matrix, row-major.
        pca_mean: [dim] numpy array mean to subtract before projection.
        pca_dims: # of dimenstions to use in output data, of type int.
        pca_vars: [dim] numpy array containing PCA variances. 
                   Only used if use_whitening is True.
        use_whitening: Whether whitening is to be used. usually recommended.
    Returns:
        output: [N, output_dim] FloatTensor with output of PCA/Whitening operation.
    (Warning: element 0 in pca_variances might produce nan/inf value.) 
    '''
    pca_mean = torch.from_numpy(pca_mean).float()
    pca_vars = torch.from_numpy(pca_vars).float()
    pca_matrix = torch.from_numpy(pca_matrix).float()

    if torch.cuda.is_available():
        pca_mean = pca_mean.cuda()
        pca_vars = pca_vars.cuda()
        pca_matrix = pca_matrix.cuda()

    data = data - pca_mean
    output = data.matmul(pca_matrix.narrow(0, 0, pca_dims).transpose(0,1))
    
    if use_whitening:
        output = output.div((pca_vars.narrow(0, 0, pca_dims) ** 0.5))
    return output
