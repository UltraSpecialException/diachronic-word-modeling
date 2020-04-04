import numpy as np


class TSNE:
    """
    t-SNE projection for dimensionality reduction.
    """

    def __init__(self, high_dim_data, perplexity=1e-5, tolerance=30.0,
                 normalize=False):
        """
        Initialize an instance of t-SNE analysis class.
        """
        self.data = high_dim_data
        self.perplexity = perplexity
        self.tolerance = tolerance
        if normalize:
            self.transform_stable()

    def transform_stable(self):
        """
        Normalize the data and then remove NaN data points.
        """
        data = self.data
        data = np.nan_to_num(
            (data - np.mean(data, axis=0)) / np.std(data, axis=0), 0)
        self.data = data

    def pairwise_distance(self, data=None):
        """
        Compute the pairwise distances of all the high dimensional data stored.
        """
        if data is None:
            data = self.data
        assert len(data.shape) == 2
        diff = data[:, None] - data
        dist = (diff ** 2).sum(axis=2)

        return dist

    @staticmethod
    def compute_probs(distance, sigma, word_pos):
        """
        Compute the probability distribution of neighbors of the word at
        <word_pos>
        """
        neg_dist = -distance / (2 * sigma ** 2)
        exp_neg_dist = np.exp(neg_dist)

        exp_neg_dist[word_pos] = 0

        probs = exp_neg_dist / exp_neg_dist.sum()
        return probs

    @staticmethod
    def compute_perplexity(probs, word_pos):
        """
        Return the perplexity of the probability distribution of neighbors
        of the word at <word_pos>.
        """
        log_probs = np.log2(probs)
        log_probs[word_pos] = 0
        return 2 ** -(probs * log_probs).sum()

    def get_sigma(self, distance, word_pos, sigmas):
        """
        Compute the appropriate sigma value for probability distribution
        computation such that the perplexity is the same for all words'
        probability distributions.
        """
        perplexity, tolerance = self.perplexity, self.tolerance
        min_val, max_val = -float("inf"), float("inf")

        iters = 0
        probs = self.compute_probs(distance, sigmas[word_pos], word_pos)
        computed_perplexity = self.compute_perplexity(probs, word_pos)
        diff = perplexity - computed_perplexity
        at_limit = False

        while abs(diff) > tolerance and iters < 50:
            if diff > 0:
                min_val = sigmas[word_pos]
                if at_limit:
                    sigmas[word_pos] = (min_val + max_val) / 2
                else:
                    max_val = sigmas[word_pos] * 2
                    sigmas[word_pos] = max_val
            else:
                max_val = sigmas[word_pos]
                sigmas[word_pos] = (min_val + max_val) / 2
                at_limit = True

            probs = self.compute_probs(distance, sigmas[word_pos], word_pos)
            computed_perplexity = self.compute_perplexity(probs, word_pos)
            diff = perplexity - computed_perplexity
            iters += 1

        return probs

    def get_probs_dists(self, data=None):
        """
        Compute the probability distribution for each word.
        """
        if data is None:
            data = self.data

        num_words = data.shape[0]

        distances = self.pairwise_distance(data)
        scaled_distances = distances / distances.std(axis=-1) * 10

        final_probs = np.zeros((num_words, num_words))
        sigmas = np.ones(num_words)

        for word_pos in range(num_words):
            distance = scaled_distances[word_pos]
            final_probs[word_pos] = self.get_sigma(distance, word_pos, sigmas)

        return final_probs

    def pca(self, data=None, target_dim=None):
        if data is None:
            data = self.data
        eigenvectors, eigenvalues = np.linalg.eig(data.T @ data)
        if target_dim:
            return np.real(data @ eigenvalues[:, :target_dim])
        else:
            variances = eigenvectors / sum(eigenvectors)

            total = 0
            for i, var in enumerate(variances):
                total += var
                if total >= 0.8:
                    return np.real(data @ eigenvalues[:, :i])

    def project(self, num_words, lr, momentum, target_dim, max_iter,
                do_pca=True, pca_dim=2):
        """
        Run tsne using gradient descent.
        """
        if do_pca:
            original_probs = self.get_probs_dists(self.pca(target_dim=pca_dim))
        else:
            original_probs = self.get_probs_dists()

        n = num_words
        new_dist = np.random.normal(loc=0, scale=0.01, size=(n, target_dim))

        v = 0
        for iter in range(max_iter):
            pairwise_new_dist = self.pairwise_distance(new_dist)
            q = 1 / (1 + pairwise_new_dist)
            np.fill_diagonal(q, 0)
            Q = q / np.sum(q, axis=1, keepdims=True)
            new_dist_flat = new_dist.flatten()
            d = new_dist_flat.reshape(target_dim, n, 1, order='F') \
                - new_dist_flat.reshape(target_dim, 1, n, order='F')

            CE = -original_probs * np.log2(Q)
            np.fill_diagonal(CE, 0)

            gd = 4 * (original_probs - Q) * q * d
            gradient = np.sum(gd, axis=2).T

            v = lr * gradient + momentum * v
            new_dist -= v

        return new_dist
