import numpy as np
from numpy.polynomial.polynomial import polypow
from collections import deque


class SARIMA_ONS:
    '''
    This class implements Seasonal ARIMA Online Newton Step algorithm
    '''
    def __init__(self, history, order, order_s, diff, diff_s, period, seed=42):
        self.order = order # AR order p
        self.order_s = order_s # seasonal AR order P
        self.diff = diff # non-seasonal difference order d
        self.diff_s = diff_s # seasonal difference order D
        self.period = period # seasonal period s
        
        # original series [X_{t-1}, X_{t-2}, ..., X_{t-(p+s*P+d+s*D)}]
        self.X = np.zeros(order + period * order_s + diff + period * diff_s)
        trunc = list(reversed(history))[:order + period * order_s + diff + period * diff_s]
        self.X[:len(trunc)] = trunc
        self.X = deque(self.X, maxlen=order + period * order_s + diff + period * diff_s)
        
        # compute (1-B)^d (1-B^s)^D
        self.diff_polycoef = polypow([1, -1], diff) # (1-B)^D
        self.diff_s_polycoef = polypow([1] + [0] * (period-1) + [-1], diff_s) # (1-B^s)^D
        self.diff_multiply = self._polymul(self.diff_polycoef, self.diff_s_polycoef)
        # differenced series [Y_{t-1}, Y_{t-2}, ..., Y_{t-(p+s*P)}]
        self.Y = self._compute_backshift(self.diff_multiply, self.X)
        assert len(self.Y) == order + period * order_s
        self.Y = deque(self.Y, maxlen=order + period * order_s)
        
        
        self.c = 1
        self.X_max = 2
        self.D = 2 * self.c * np.sqrt(order + order_s)
        self.G = self.D * (self.X_max**2)
        self.lambda_ = 1.0 / (order + order_s) if order + order_s != 0 else 1.0
        self.eta = 0.5 * min(4 * self.G * self.D, self.lambda_) # learning rate
        self.epsilon = 1.0 / (self.eta * self.D)**2
        self.A = np.matrix(np.diag([1] * (order + order_s)) * self.epsilon) # hessian matrix
        if seed:
            np.random.seed(seed)
        self.gamma = np.matrix(np.random.uniform(-self.c, self.c, (order, 1))) # parameters g_1, g_2, ..., g_p
        if seed:
            np.random.seed(seed)
        self.gamma_s = np.matrix(np.random.uniform(-self.c, self.c, (order_s, 1))) # seasonal parameters gs_1, gs_2, ..., gs_P
        
        
    def _polymul(self, a1, a2):
        '''
        Find the product of two polynomials. 
        Each input must be a 1D sequence of polynomial coefficients, from LOWEST to HIGHEST degree.
        The result is also a 1D sequence of polynomial coefficients, from LOWEST to HIGHEST degree.
        '''
        a1_flip = np.flip(a1)
        a2_flip = np.flip(a2)
        return np.flip(np.polymul(a1_flip, a2_flip))
    
    def _compute_backshift(self, polycoef, X, sliding=True):
        '''
        Compute f(B) X_t where f is a polynomial function
        
        params:
         polycoef = [a_0, a_1, a_2, ..., a_k], f(B) = a_0 + a_1 * B + ... + a_k * B^k (len = k + 1)
         X = [X_t, X_{t-1}, ..., X_{t-n}] (len = n + 1)
        returns:
         if sliding is True returns [f(B) X_t, f(B) X_{t-1}, ..., f(B) X_{t-n+k}] (len = n - k + 1) 
         else returns only the first element f(B) X_t.
        '''
        k = len(polycoef) - 1
        n = len(X) - 1
        X = list(X)
        if sliding:
            return [np.dot(polycoef, X[i:i + k + 1]) for i in range(n - k + 1)]
        else:
            return np.dot(polycoef, X[:k + 1])
        
        
    def predict(self, gamma_polycoef, gamma_s_polycoef):
        '''
        Y_pred_t = gamma_polynomial(B) Y_t + gamma_s_polynomial(B^s) Y_t - gamma_polynomial(B) * gamma_s_polynomial(B^s) Y_t
        where
         gamma_polynomial(B) = gamma_1 * B + gamma_2 * B^2 + ... + gamma_p * B^p
         gamma_s_polynomial(B^s) = gamma_s_1 * B^s + gamma_s_2 * B^(2s) + ... + gamma_s_P * B^(Ps)
         
        params:
         gamma_polycoef = [gamma_1, gamma_2, ..., gamma_p]
         gamma_s_polycoef = [[0]*(s-1), gamma_s_1, [0]*(s-1), gamma_s_2, ... , [0]*(s-1), gamma_s_P]
        '''
        gamma_polycoef = [0] + gamma_polycoef
        gamma_s_polycoef = [0] + gamma_s_polycoef
        
        Y_pred1 = self._compute_backshift(gamma_polycoef, [0] + list(self.Y), sliding=False)
        Y_pred2 = self._compute_backshift(gamma_s_polycoef, [0] + list(self.Y), sliding=False)
        
        Y_pred3 = self._compute_backshift(self._polymul(gamma_polycoef, gamma_s_polycoef), [0] + list(self.Y), sliding=False)
        return Y_pred1 + Y_pred2 - Y_pred3
    
    
    def update_parameters(self, y, y_pred, gamma_polycoef, gamma_s_polycoef):
        '''
        delta(loss_t)/delta(gamma_i) = -2 * (Y_t - Y_pred_t) * [B^i * gamma_s_polynomial(B^s)](Y_t), i=1,...,p
        delta(loss_t)/delta(gamma_s_j) = -2 * (Y_t - Y_pred_t) * [B^(js) * gamma_polynomial(B)](Y_t), j=1,...,P
        where
         gamma_polynomial(B) = 1 - gamma_1 * B - gamma_2 * B^2 - ... - gamma_p * B^p
         gamma_s_polynomial(B^s) = 1 - gamma_s_1 * B^s - gamma_s_2 * B^(2s) - ... - gamma_s_P * B^(Ps)
        
        params:
         gamma_polycoef = [gamma_1, gamma_2, ..., gamma_p]
         gamma_s_polycoef = [[0]*(s-1), gamma_s_1, [0]*(s-1), gamma_s_2, ... , [0]*(s-1), gamma_s_P]
        '''
        gamma_polycoef = [1] + list(-np.array(gamma_polycoef))
        gamma_s_polycoef = [1] + list(-np.array(gamma_s_polycoef))
        
        # compute gradient w.r.t gamma and gamma_s
        polynomial_list = [[0]*i + gamma_s_polycoef for i in range(1, self.order+1)]
        nabla = [self._compute_backshift(polynomial, [y] + list(self.Y), sliding=False) for polynomial in polynomial_list]
        
        polynomial_s_list = [[0]*(i*self.period) + gamma_polycoef for i in range(1, self.order_s+1)]
        nabla_s = [self._compute_backshift(polynomial_s, [y] + list(self.Y), sliding=False) for polynomial_s in polynomial_s_list]
        
        nabla_all = -2 * (y - y_pred) * np.array(nabla + nabla_s)
        
        # reshape
        nabla_all = nabla_all.reshape(-1,1)
        
        # update parameters
        self.A += np.dot(nabla_all, nabla_all.T)
        grad = 1 / self.eta * np.dot(np.linalg.inv(self.A), nabla_all)
        self.gamma -= grad[:self.order]
        self.gamma_s -= grad[self.order:]
    
    
    def update_history(self, x, y):
        self.X.appendleft(x)
        self.Y.appendleft(y)
        
    
    def fit_one_step(self, x):
        '''
        Run one iteration of the algorithm
        
        params:
         x: float, observation value at t (X_t)
        
        returns:
         x_pred: float, prediction value at t (Xpred_t)
         loss: float, squared loss (x - x_pred)^2
        '''
        gamma_polycoef = list(np.array(self.gamma).squeeze(-1)) # [g_1, ..., g_p]
        gamma_s_polycoef = list(np.array(self.gamma_s).squeeze(-1)) # [gs_1, ..., gs_P]
        if self.order_s != 0: # zero check
            gamma_s_polycoef = list(np.kron(gamma_s_polycoef, [0]*(self.period-1) + [1])) # [[0]*(s-1), gs_1, [0]*(s-1), gs_2, ..., gs_P]
        
        # predict
        y_pred = self.predict(gamma_polycoef, gamma_s_polycoef)
        x_pred = y_pred - self._compute_backshift(self.diff_multiply, [0] + list(self.X), sliding=False)
        
        # observe actual y
        y = self._compute_backshift(self.diff_multiply, [x] + list(self.X), sliding=False)
        
        # compute loss
        loss = (x - x_pred)**2
        
        # update parameters
        self.update_parameters(y, y_pred, gamma_polycoef, gamma_s_polycoef)
        
        # update history
        self.update_history(x, y)
        
        return x_pred, loss
    
    
def mape(true, pred):
    return np.mean(np.abs((true - pred) / true))


def rmse(true, pred):
    return np.sqrt(np.mean((true - pred)**2))


def mae(true, pred):
    return np.mean(np.abs(true - pred))