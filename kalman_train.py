import numpy as np

T0 = 0
DELTA_T = .2
PROCESS_RNG_SEED = None
MEASUREMENT_RNG_SEED = None

class Process:
    def __init__(self,x0,v0,sigma_a):
        """
        x0 - initial postion (m)
        v0 - initial velocity (m/s)
        sigma_a - standard deviation of acceleration for random accelerations at each time step (m/s/s)
        """
        #private
        self._rng = np.random.default_rng(PROCESS_RNG_SEED)
        self._sigma_a = sigma_a
        
        #public
        self.x = x0
        self.v = v0
        self.t = T0
        
    def _draw_random_acceleration(self):
        unmodeled_a = self._rng.normal(np.sin(self.t),self._sigma_a/10)
        return self._rng.normal(0,self._sigma_a)+unmodeled_a
        
    def update(self):
        a = self._draw_random_acceleration()
        self.x = self.x + self.v*DELTA_T + 1/2*a*DELTA_T**2
        self.v = self.v + a*DELTA_T
        self.t += DELTA_T
        
class Measurement:

    def __init__(self,process,sigma_z,drop_range=None):
        """
        process - Process instance to simulate measurements of
        sigma_z - position measurement uncertainty
        drop_range - set to list [t_start,t_end] to return measurement None between these times
        """
        #private
        self._rng = np.random.default_rng(MEASUREMENT_RNG_SEED)
        self._process = process
        self._sigma_z = sigma_z
        self._range = drop_range
        #public
        self.t = T0
        self.z = None
        
    def update(self):
        if self._range is not None and self.t > self._range[0] and self.t < self._range[1]:
            self.z = None
        else:
            self.z = self._rng.normal(self._process.x,self._sigma_z)
        self.t += DELTA_T
        
class Filter:
    def __init__(self,x0,v0,sigma_a,sigma_z,Sigma0=None):
        """
        x0 - float
            initial postion (m)
        v0 - float
            initial velocity (m/s)
        sigma_a - float
            standard deviation of acceleration for process noise model (Q matrix) (m/s/s)
        sigma_z - float
            position measurement uncertainty (for measurment covariance matrix)
        Sigma0 - 2x2 array, optional
            Initial filter covariance "guess", if not set will init to 2 * process covariance 
        """
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        
        # Variables from filter equations
        self.A = np.array([[1,DELTA_T],
                           [0,1]])
        self.R = np.array([self.sigma_z**2]) # 1 x 1 matrix
        self.Q = np.array([[1/4*DELTA_T**4,1/2*DELTA_T**3],
                           [1/2*DELTA_T**3,DELTA_T*2]])*self.sigma_a**2 # 2 x 2 matrix
        self.C = np.array([[1,0]]) # 1 x 2 row matrix

        #Filter state (intialize using constructor inputs)
        self.mu = np.array([[x0],[v0]]) # 2 x 1 column matrix
        self.Sigma = self.Q*2 if Sigma0 is None else Sigma0 # intialize filter covariance to process covariance if unset
        self.t = T0
        
    def predict(self):
        mu_bar = self.A @ self.mu
        Sigma_bar = self.A @ (self.Sigma @ self.A.T) + self.Q
        return mu_bar, Sigma_bar

    def update(self,z):
        """
        z - float
            measurement of position
        """
        mu_bar,Sigma_bar = self.predict()
        if z is not None:
            K_denominator = self.C @ (Sigma_bar @ self.C.T) + self.R
            K = Sigma_bar @ (self.C.T @ np.linalg.inv(K_denominator))
            I = np.eye(self.mu.size)
            self.mu = mu_bar + K @ ( z - self.C @ mu_bar)
            self.Sigma = (I - K @ self.C) @ Sigma_bar
        else:
            self.mu = mu_bar
            self.Sigma = Sigma_bar
        self.t += DELTA_T