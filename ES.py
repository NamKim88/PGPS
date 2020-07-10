import numpy as np

class CEM:
    def __init__(self, args, num_params, mu_init=None):
        # Params inform.
        self.num_params = num_params
        self.mu = np.zeros(self.num_params) if (mu_init is None) else np.array(mu_init)

        # Dist the inform.
        self.pop_size = args.pop_size              
        self.cov = args.cov_init * np.ones(self.num_params)
        self.cov_limit = args.cov_limit
        self.damp = args.cov_init
        self.alpha = args.cov_alpha        

        # Elitism
        self.elitism = args.elitism
        self.elite_param = self.mu

        # Parents and weights
        self.parents = (args.pop_size // 2)
        self.weights = np.array([np.log((self.parents + 1) / i) for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

        # Print the inform.
        print(f"\nThe weight of CEM update: {np.round(self.weights, 3)}")
        print(f"ES elitism: {self.elitism}")

    def sampling(self, pop_size):
        # Sample perturbation from N(0, np.sqrt(cov))
        epsilon = np.sqrt(self.cov) * np.random.randn(pop_size, self.num_params)

        # Generate policy individuals
        inds = self.mu + epsilon

        # If use elitism, change last individual as previous elite individual
        if self.elitism:
            inds[0] = self.elite_param

        return inds

    def update(self, params, scores):
        # Get index following ascending scores order, and determine parents
        idx_sorted = np.argsort(-np.array(scores))
        params_parents = params[idx_sorted[:self.parents]]

        # Update cov
        z = (params_parents - self.mu)
        self.damp = (1 - self.alpha) * self.damp + self.alpha * self.cov_limit
        self.cov = 1 / self.parents * self.weights @ (z * z) + self.damp * np.ones(self.num_params)

        # Update mu
        self.mu = self.weights @ params_parents

        # Save elite individual
        self.elite_param = params[idx_sorted[0]]

        print(f"Mu:{np.round(self.mu[:5], 4)}")
        print(f"Cov:{np.round(self.cov[:3], 6)}")
        

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def set_cov(self, cov):
        self.cov = np.array(cov)

    def get_distrib_params(self):
        return np.copy(self.mu), np.copy(self.cov)