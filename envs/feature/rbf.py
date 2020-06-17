import numpy as np
"""
Isotropic Gaussian RBF Features
"""

class GaussianRBF:

    def __init__(self, mean, variance, K=2, dims=2):
        """
        :param mean: (np.ndarray) mean vector Kxdim
        :param variance: (np.ndarray)  variance vector Kx1
        :param K: number of basis functions
        :param dims: dimension of the input
        """
        assert mean.shape == (K, dims)
        #assert variance.shape == (K, )

        self._K = K
        self._mean = mean
        self._dims = dims
        self._var = variance

    def _compute(self, point):

        """
        Computes a feature vector for the point given
        :param point: np.ndarray (dim)
        :return: feature vector: np.ndarray
        """
        val = []

        for k in range(self._K):
            dif = self._mean[k, :] - point
            val.append(np.exp(-1/2 * np.dot(dif / self._var[k], dif)))
        f = np.asarray(val, order='F')
        f = f/np.sum(f)
        return f

    def __call__(self, x):
        if x.ndim == 2:
            return self._compute_feature_matrix(x)
        elif x.ndim == 1:
            return self._compute(x)

    def _compute_feature_matrix(self, data):
        """
        Computes the feature matrix for the dataset passed
        :param data: np.ndarray with a sample per row
        :return: feature matrix (np.ndarray) with feature vector for each row.
        """
        assert data.shape[1] == self._dims
        features = []
        for x in range(data.shape[0]):
            features.append(self._compute(data[x, :]))

        return np.asarray(features)

    def number_of_features(self):
        return self._K


def build_features_mch_state(mch_size, n_basis, state_dim):
    """Create RBF for mountain car"""
    # Number of features
    K = n_basis[0] * n_basis[1]
    # Build the features
    positions = np.linspace(mch_size['position'][0], mch_size['position'][1], n_basis[0])
    speeds = np.linspace(mch_size['speed'][0], mch_size['speed'][1], n_basis[1])
    mean_positions, mean_speeds = np.meshgrid(positions, speeds)
    mean = np.hstack((mean_positions.reshape(K, 1), mean_speeds.reshape(K, 1)))

    position_size = mch_size['position'][1] - mch_size['position'][0]
    positions_var = (position_size / (n_basis[0] - 1) / 2) ** 2
    speed_size = mch_size['speed'][1] - mch_size['speed'][0]
    speeds_var = (speed_size / (n_basis[1] - 1) / 2) ** 2
    var = np.array([np.tile(positions_var, K), np.tile(speeds_var, K)]).T

    return GaussianRBF(mean, var, K=K, dims=state_dim)

def build_features_gw_state(gw_size, n_basis, state_dim):
    """Create RBF for gridworld as functions of the state"""
    # Number of features
    K = n_basis[0] * n_basis[1]
    # Build the features
    x = np.linspace(0, gw_size[0], n_basis[0])
    y = np.linspace(0, gw_size[1], n_basis[1])
    mean_x, mean_y = np.meshgrid(x, y)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1)))

    state_var = (gw_size[0] / (n_basis[0] - 1) / 2) ** 2
    var = np.tile(state_var, (K))

    return GaussianRBF(mean, var, K=K, dims=state_dim)


if __name__ == '__main__':
    mean = np.array([[1, 2], [3, 4]])
    var = np.array([3, 8])

    rbf = GaussianRBF(mean, var)

    data = np.random.random((100, 2))

    matrix = rbf(data)
    print(matrix)
    print(rbf(np.array([1, 2])))