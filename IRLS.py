import numpy as np

class IRLS:
    def __init__(
        self, 
        fit_intercept = "column",
        interactions=None,
        verbose_algorithm = False,
        iteration_limit = 100
    ):
        self.fit_intercept = fit_intercept
        self.interactions = interactions
        self.verbose_algorithm = verbose_algorithm
        self.iteration_limit = iteration_limit
    
    @staticmethod
    def _logit(x):
        return np.log(x/(1-x))
    
    @staticmethod
    def _ilogit(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _log_likelihood(p, y):
        # More numerically stable log likelihood
        L = y * np.log(p) + (1 - y) * np.log(1 - p)
        return sum(L)
    
    @staticmethod
    def _weighted_least_squares(X, W, z):
        # (X′WX)^−1 X′Wz
        hessian =  X.T @ W @ X
        grad = X.T @ W @ z
        hessian_inv = np.linalg.inv(hessian)
        weighted_least_squares = hessian_inv @ grad
        return weighted_least_squares 
    
    def _add_interactions(self, X):
        X = np.array(X)
        interacted_columns = np.array([
            X[:, i] * X[:, j]
            for i, j in self.interactions
        ]).T
        return np.hstack((X, interacted_columns))
    
    def _prepare_data(self, X):
        if self.interactions:
            X = self._add_interactions(X)
        if self.fit_intercept == "column":
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X
    
    def _algorithm(self, X, y):
        # 
        self.beta = np.zeros(self.total_features)
        self.beta_zero = self._logit(np.mean(y)) if self.fit_intercept == "prior" else 0.0
        old_ll = 0
        
        for iteration in range(1, self.iteration_limit + 1):
            eta        = self.beta_zero + self.beta @ X.T
            mu         = self._ilogit(eta)
            s          = mu * (1 - mu)
            z          = eta + (y - mu) / s
            S          = np.diag(s)
            self.beta  = self._weighted_least_squares(X, S, z)
            
            ll         = self._log_likelihood(mu, y)
            if self.verbose_algorithm:
                print(f"Iteration {iteration}, log likelihood: {ll}")
            if abs(ll - old_ll) < 1e-12:
                break
            
            old_ll = ll
        
        self.fitted_log_likelihood = ll
        self.fitted_iterations = iteration
    
    def _data_asserts(self, X, y):
        assert X.shape[0] > X.shape[1], (
            f"Number of observations should be higher than number of features. {X.shape[0]} > {X.shape[1]}")
        assert X.shape[0] == y.shape[0], f"X and y are of different lengths. {X.shape[0]} != {y.shape[0]}"
        
    
    def fit(self, X, y, interactions=None):
        _, self.real_features = X.shape
        self._data_asserts(X, y)
        
        if interactions is not None:
            self.interactions = interactions
        
        X =  self._prepare_data(X)
        _, self.total_features = X.shape
        self._algorithm(X, y)
        
        return self
    
    def predict_proba(self, X):
        X =  self._prepare_data(X)
        return self._ilogit(self.beta_zero + self.beta @ X.T)
    
    def predict(self, X, cutoff=0.5):
        return (self.predict_proba(X) > cutoff) + 0
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
    
    def params(self, which=None):
        
        if self.fit_intercept == "column":
            intercept = self.beta[0] 
            beta_features = self.beta[1:] 
            full_beta = self.beta
        else:
            full_beta = np.concatenate(([self.beta_zero], self.beta))
            intercept = self.beta_zero
            beta_features = self.beta
        
        if self.interactions:
            beta_interactions = beta_features[1+len(self.interactions):] 
            beta_features = beta_features[:-len(self.interactions)] 
        else:
            beta_interactions = np.array([])
        
        param_dict = {
            "beta": full_beta,
            "intercept": intercept, 
            "beta_features": beta_features,
            "beta_interactions": beta_interactions,
            "fitted_log_likelihood": self.fitted_log_likelihood,
            "fitted_iterations": self.fitted_iterations
        }
        
        if which is None:
            return param_dict
        else:
            return param_dict[which]