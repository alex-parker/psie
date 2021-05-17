import numpy as np 

def ks(x1, x2):
    data1 = np.sort(x1)
    data2 = np.sort(x2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]    
    
    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2
    minS = np.clip(-np.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = np.max(cddiffs)
    return max(minS, maxS)

def ad(x1, x2):
    k = 2
    n = np.array([x1.size, x2.size])

    Z = np.sort(np.hstack((x1, x2)))
    N = Z.size
    Zstar = np.unique(Z)
 
    A2akN = 0.
    Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    if N == Zstar.size:
        lj = 1.
    else:
        lj = Z.searchsorted(Zstar, 'right') - Z_ssorted_left
        
    Bj = Z_ssorted_left + lj / 2.
    s = np.sort(x1)
    s_ssorted_right = s.searchsorted(Zstar, side='right')
    Mij = s_ssorted_right.astype(float)
    fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
    Mij -= fij / 2.
    inner = lj / float(N) * (N*Mij - Bj*n[0])**2 / (Bj*(N - Bj) - N*lj/4.)
    A2akN += inner.sum() / n[0]

    s = np.sort(x2)
    s_ssorted_right = s.searchsorted(Zstar, side='right')
    Mij = s_ssorted_right.astype(float)
    fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
    Mij -= fij / 2.
    inner = lj / float(N) * (N*Mij - Bj*n[1])**2 / (Bj*(N - Bj) - N*lj/4.)
    A2akN += inner.sum() / n[1]    
    
    A2akN *= (N - 1.) / N
    return A2akN

def ku(x1, x2):
    data1 = np.sort(x1)
    data2 = np.sort(x2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]    
    
    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2
    
    minS = np.clip(-np.min(cddiffs), 0, 1)  # Ensure sign of minS is not negative.
    maxS = np.max(cddiffs)
    
    return minS + maxS