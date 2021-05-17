import numpy as np 
import importlib.util
from . import metrics
from tqdm import tqdm, trange

class ABCr:
    '''
    Generic class for size-frequency distribution manipulation, 
    observational filtering, and likelihood-free inference.
    '''
    def __init__(self, model=None, priors={}):

        self._pp = priors.copy()
        self._priors()
        self._model = model
        self.sample = []
        self.tested_params = []
        self.test_values = []
        self._bad_parameter_flag = False
        self._data = None
        self._n_iter = 1
        self.pdf = None

        return None

    
    def iterate(self, data=None, n_iter=1, seed=None):

        if seed != None:
            np.random.seed(seed)

        for i in trange(0, n_iter):
            self.sample_new_parameters()
            self.generate_sample(N=data.shape[-1])
            if self._bad_parameter_flag:
                self.tested_params.append(self._p)
                self.test_values.append(np.inf)
            else:
                self.measure_distance_metric(data)

        return self
    

    def iterate_parallel(self, data=None, n_iter=1, pool=None):

        self._data = data.copy()
        self._n_iter = n_iter

        pool_size = len(pool._pool) 

        _jobs = []
        for i in range(1, 1 + pool_size):
            _sub = self.fresh_copy()
            results = pool.apply_async(_sub.iterate, args=(data, n_iter, i))
            _jobs.append(results)
        for j in _jobs:
            j.wait()
            k = j.get()
            for par in k.tested_params:
                self.tested_params.append(par)
            for val in k.test_values:
                self.test_values.append(val)

    def sample_pdf(self, thresh=0.1):
        ok = np.where(np.array(self.test_values) <= np.percentile(self.test_values, thresh))
        keep_params = np.array(self.tested_params)[ok]
        sample_dict = {}

        for k in self._pp.keys():
            sample_dict[k] = np.zeros((keep_params.shape[0]))

        for i in range(0, keep_params.shape[0]):
            for k in self._pp.keys():
                sample_dict[k][i] = keep_params[i][k] 

        self.pdf = sample_dict.copy()
        return sample_dict

    def generate_sample(self, N=0):
        self.sample, self._bad_parameter_flag = self._model(self._p, N=N)
        return None    
    
    def sample_new_parameters(self, ):
        self._priors()
        return None
    
    def measure_distance_metric(self, data, metric='AD'):
        
        ### simple 1D cases; need to look at properties of 'data' to 
        ### catch this case
        D_sum = 0
        if metric=='AD':
            for i in range(data.shape[0]):
                D_sum += metrics.ad(self.sample[i], data[i])
                #print(data[i].shape, self.sample[i].shape)
        elif metric=='KS':
            for i in range(data.shape[0]):
                D_sum += metrics.ks(self.sample[i], data[i])
        elif metric=='Kuipers':
            for i in range(data.shape[0]):
                D_sum += metrics.ku(self.sample[i], data[i])

        else:
            ### raise an error here, eventually...
            print('test not recognized')

        self.tested_params.append(self._p)
        self.test_values.append(D_sum)
        
        
        ### cases to add:
        ### data and self.sample are dicts
        ### and some metrics compute distances over specific
        ### parameters (keys) in those dicts.
        ### For example: orbital_distance is computed on (a,e,i)
        ### data = {'a':[0.7, 1.0, 1.5], 'e':[0.1, 0.01, 0.15], 'i':[0.1, 0.01, 0.15]}
        ### orbital_distance(data)=D
        ###
        
        return None
    
    def _precompute(self, ):
        ### If model state can be broken into slow and fast calculations,
        ### 'precompute' can be done to set slow state of model,
        ### then generate_sample can be called multiple times to 
        ### compute the fast sampling step.
        
        return None
    
    def _priors(self, ):
        _p1 = {}
        for key in self._pp.keys():
            ### dict structure: 'key': type, ...
            ### if type=='callable':
            ### --> new_param = self._parameter_priors(key)
            ###     TODO: ADD PASSABLE ARGS + KWARGS
            ### if type=='uniform_float':
            ### --> sample from np.random.uniform, val1 to val2
            ### if type=='uniform_int':
            ### --> sample from np.random.randint, val1 to val2
            
            if self._pp[key][0] == 'callable':
                _p1[key] = self._pp[key][1]()
            elif self._pp[key][0] == 'uniform_float':
                _p1[key] = np.random.uniform( self._pp[key][1], self._pp[key][2] )
            elif self._pp[key][0] == 'uniform_int':
                _p1[key] = np.random.randint( self._pp[key][1], self._pp[key][2] )
            elif self._pp[key][0] == 'uniform_log10':
                _p1[key] = 10**np.random.uniform( self._pp[key][1], self._pp[key][2] )
            elif self._pp[key][0] == 'uniform_ln':
                _p1[key] = np.exp(np.random.uniform( self._pp[key][1], self._pp[key][2] ))
            elif self._pp[key][0] == 'uniform_log2':
                _p1[key] = 2**np.random.uniform( self._pp[key][1], self._pp[key][2] )
            else:
                ### pass old value?
                _p1[key] = self._p[key]
                
        self._p = _p1.copy()
        return None
    

    def _filters(self, ):
        return None

    def fresh_copy(self):
        return ABCr(model=self._model, priors=self._pp)

    def burn_in_copy(self):
        abc = ABCr(model=self._model, priors=self._pp)
        abc.tested_params = self.tested_params.copy()
        abc.test_values = self.test_values.copy()
        return abc
    
