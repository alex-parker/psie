import numpy as np 

class craterscaling:
    def __init__(self, a=0.217, b=0.333, c=0.783, 
                 vesc=1.2, g=62.0, zeta=0.108, D_transition=10.0,
                 delta=0.6, rho=1.0, scale=13.4):

        ### all km, km/s except for g; g is in cm/s^2
        self.scale = scale
        self.a = a
        self.b = b
        self.c = c
        self.vesc = vesc
        self.g = g
        self.zeta = zeta
        self.delta = delta ### density of impactor
        self.rho = rho     ### density of target
        self.D_transition = D_transition

    def impactor_to_crater(self, d_impactor_km, v_impactor_kms, theta_impact_rad):
        diams = d2D_v1(d_impactor_km, v_impactor_kms, theta_impact_rad, a=self.a, b=self.b, c=self.c, 
                 vesc=self.vesc, g=self.g, zeta=self.zeta, D_transition=self.D_transition,
                 delta=self.delta, rho=self.rho, scale=self.scale)
        return diams

def d2D_v1(d_impactor_km=np.empty(shape=0, dtype=float), 
           v_impactor_kms=np.empty(shape=0, dtype=float), 
           theta_impact_rad=np.empty(shape=0, dtype=float), a=0.217, b=0.333, c=0.783, 
                 vesc=1.2, g=62.0, zeta=0.108, D_transition=10.0,
                 delta=0.6, rho=1.0, scale=13.4):

        vfull2    = (v_impactor_kms**2 + vesc**2)
        term1     = (vfull2 / g)**a
        term2     = (delta / rho)**b
        term3     = d_impactor_km**c
        D         = scale * term1 * term2 * term3 * np.cos(theta_impact_rad)**(1.0/3.0)
        D_complex = D * (D/D_transition)**( zeta )

        cmplx     = D > D_transition
        D[cmplx]  = D_complex[cmplx] 

        return D

def broken_power_law(a1=2.0, a2=4.0, xb_in=1.0, N=1000, xmin=0.0):

    ### a1: index for power-law to left of break
    ### a2: index for power-law to right of break
    ### xb: break location
    ### N:  final sample size

    xb = xb_in - xmin

    ### CDF at break for both power-laws
    n1 = (1-(1+xb)**-a1)
    n2 = (1-(1+xb)**-a2)
    
    ### PDF at break for both power-laws after truncation
    p1 = (a1/(1+xb)**a1) / n1
    p2 = (a2/(1+xb)**a2) / (1.0 - n2)

    ### generate sample size for each power-law segment
    N1 = int(np.random.binomial(N, p2/(p1+p2)))
    N2 = int(N - N1)

    ### sample 1; use inverse CDF from 0 to xb
    #if N1 > 0:
    f11, f12 = 0.0, (1 - (1+xb)**-a1)
    u1 = np.random.uniform(f11, f12, N1)
    x1 = (1 - u1)**(-1.0/a1) - 1

    ### sample 2: use inverse CDF from xb to +inf
    f21, f22 = (1.0 - (1.0+xb)**-a2), 1.0
    u2 = np.random.uniform(f21, f22, N2)
    x2 = (1.0 - u2)**(-1.0/a2) - 1.0
        
    ### BUG HERE FOR REALLY STEEP SLOPES;
    ### RANGE IS TINY AND CLOSE TO 1.0, RESULTING IN DIV BY 0 ERRORS
    ### IN NEXT LINE.

    ### merge samples and return
    return np.hstack((x1, x2)) + xmin

def piecewise_power_law(    qv_in=np.array([2.0, 3.0, 5.0]), 
                            Db_in=np.array([1.0, 10.0, 100.0]), 
                            maxd=1000.0, N=1):
    ### requires kwargs N (sample size), mind (minimum diameter)
    ### maxd (maximum diameter), slope (power law slope alpha)
    ### returns a list of N diameters

    inds = np.argsort(Db_in)
    qv = qv_in[inds]
    xb = Db_in[inds]
    
    ### assemble CDF
    xv = np.concatenate( (10**np.arange(np.log10(xb[0]), np.log10(maxd), 0.01), xb) )
    xv.sort()
    yv = np.zeros_like(xv)
    yv[xv <= xb[1]] = (xv[xv <= xb[1]]/xb[0])**(-qv[0])
    scale = 1.0
    for i in range(1, len(qv)):
        scale *= (xb[i]/xb[0])**(qv[i]-qv[i-1])
        yv[xv>xb[i]] = scale * (xv[xv>xb[i]]/xb[0])**(-qv[i])

    delta = np.diff(xv[::-1])
    cdf = abs( np.cumsum(delta * (yv[::-1][1:] + yv[::-1][:-1]) / 2.0) )
    cdf /= cdf.max()
    cdf = np.insert(cdf, 0, 0.0)
    D = np.interp( np.random.uniform(0, 1, N), cdf, xv[::-1])
    return D




if __name__ == "__main__":
    ### run some test cases, maybe?
    csl =  craterscaling()
    D = sample_broken(-0.5, 5.0, 2.0, 1000, xmin=1.0)
    diams = csl.impactor_to_crater(D, np.random.uniform(1.0, 2.0, 1000), np.arccos(np.random.uniform(0, 1, 1000)**0.5))

    for i in range(0, 10000):
        D = sample_broken(-0.5, 5.0, 2.0, 1000, xmin=1.0)
        diams = csl.impactor_to_crater(D, np.random.uniform(1.0, 2.0, 1000), np.arccos(np.random.uniform(0, 1, 1000)**0.5))


