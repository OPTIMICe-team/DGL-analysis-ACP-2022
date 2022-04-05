import numpy as np
from scipy.integrate import simps
from scipy.special import gamma
from scipy.optimize import minimize_scalar

from snowScatt.instrumentSimulator import PSD


def _loglog_interp(x, y):
    """ Interpolator of points distributed according a power-law function
    Uses the loglog to linear analogy
    
    Parameters
    ----------
        x : array float
            abscissas, in mD relations would be max dimenstion
        y : array float
            ordinates, in mD relations would be mass
    Returns
    -------
        b : float
            exponent of the power-law relation [dimensionless]
        a : float
            prefactor of the power-law relation [ [y][x]**-b ]
        fitFunc - function
            scalar function that returns a*x**b
    """
    
    b, a = np.polyfit(np.log10(x), np.log10(y), deg=1)
    return b, 10**a, lambda x : 10**a * x**b


def _analyticLambda(N, q, a, b, mu):
    """ Derivation of Lambda from the analytic form of 3 parameters Gamma PSD
    with domain extending from 0 to infinity
    
    Parameters
    ----------
        N : float
            number concetration. Equals M_0 of the PSD
        q : float
            mass mixing ratio. Equals a*M_b of the PSD assuming a pow-law mD
        a : float
            prefactor of the m = a*D**b relation
        b : float
            exponent of the m = a*D**b relation
        mu : float
            shape parameter of the 3 parameters Gamma PSD
    Returns
    -------
        Lambda : float
            scale parameter Lambda from the analytic, infinite form
    """
    return ( (N*a/q) * (gamma(mu+b+1)/gamma(mu+1)) )**(1.0/b)


class PSDGamma3(PSD.UnnormalizedGammaPSD):
    """ 3 Parameters (N0, scale Lambda, shape mu) Unnormalized Gamma PSD
    Derived class from Gamma particle size distribution (PSD).
    It initiate a PSD.UnnormalizedGammaPSD from mass mixing ratio q
    and number concentration N
    It also requires as an argument a callable function that provides
    the mass-size relation m(D)

    Callable class to provide an gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * D**mu * exp(-Lambda*D)
    """

    
    def __init__(self, q, N, mD, mu=0.0, maxD=None, minD=1.0e-6, numPoints=1000, prec=1.0e-6, maxIter=100):
        """ Initialize 3 parameters Gamma PSD with q and N
        This function uses q, N, mu and the mD relation to calculate analytic (infinite domain) N0, Lambda (and mu)
        Then it optimizes the Lambda and N0 paramters with a minimization process, calculating the
        actual integrals N and q from the bounded [minD, maxD] domain.
        Other optional parameters control the precision of the integrals
        
        WARNING!!
        The minimization process does not work well when minD is too high, 
        better to keep it really low and extrapolate down 0 whatever function 
        you need to integrate over the PSD!!!
        
        Parameters
        ----------
        q : float
            mass mixing ratio [kg m**-3]
        N : float
            number concentration [m**-3]
        mD(float) : returns float
            scalar callable function returns mass-size relation [kg]
        mu : float
            shape parameter of the gamma distribution (defaults to 0, inverse exponential)
            for stability purposes it must not be a negative integer smaller than -1 (check this)
        maxD : float
            cutoff maximum dimension [m], defaults to 10*mD**-1(q/N) = 5*Dm (mean mass diameter)
        minD : float
            cutoff minimum dimension [m], defaults to 1.0e-6
        numPoints : int
            number of points dividing the interval [minD, maxD) to be used for integration,
            defaults to 1000
        prec : float
            Precision in the reconstruction of q and N with the estimated parameters
            defaults to 1.0e-6
        maxIter : int
            number of maximum iterations of the optimization of N0 and Lambda
            Set to default 100, already quite much cause it should converge at first iteration
            but perhaps someone wants to see how far can we push it, maybe for debugging purposes
        """
        
        self.N = N
        self.q = q
        
        # Calculate maxD if needed
        if maxD is None:
            res = minimize_scalar(lambda x: np.abs(mD(x)-q/N), bounds=[0.0, 0.2], method='bounded')
            self.maxD = 10*res.x
        else:
            self.maxD = maxD
        self.minD = minD
        self._diams = np.linspace(self.minD, self.maxD, numPoints)
        self.mD = mD
        
        # first estimate mD with a power law
        b, a, _ = _loglog_interp(self._diams, self.mD(self._diams))
        Lambda_0 = _analyticLambda(N, q, a, b, mu)
        
        # first init
        PSD.UnnormalizedGammaPSD.__init__(self, 1.0, Lambda_0, mu, self.maxD)
        self.N0 *= self.N/self.calc_N()
        
        nIter = 0
        while (np.abs(self.calc_q()-self.q)/self.q>prec): # need condition only on q cause N is always perfect
            self._optimize_Lambda_N0()
            nIter += 1
            if nIter > maxIter:
                print(f'Warning, convergence is taking too long, stopped at {np.abs(self.calc_q()-self.q)/self.q}')
                break
    
    
    def __call__(self, D):
        """ Value of the PSD at specific D
        Returns PSD(D) per unit of D, not PSD(D)dD
        
        Parameters
        ----------
            D : array-like(float)
                the diameters at which the PSD(D) must be computed.
                works also with scalar values
        Returns
        -------
            psd : array-like(float)
                the value of the PSD at D. Returns scalar if D is scalar
        """
        # Just add a return 0 for minimum value
        psd = PSD.UnnormalizedGammaPSD.__call__(self, D)
        psd[(D < self.minD)] = 0.0 # the case D == 0 should be ok already
        return psd
    
    
    def integrate(self, func):
        """ Integrate function func over the PSD using the Simpson rule
        It can be used to compute any relevant integral provided that a suitable
        function of the PSD diameters is found. As an example a function that
        returns always 1 would be integrated to the total N concentration. The integral
        of the mD relation would give the mass mixing ratio and the integral of the
        backscattering cross-section would return the reflectivity.
        
        Parameters
        ----------
        func : scalar function
            A valid (possibly real) function to be integrated over the PSD
        Returns
        -------
        integral : float
            The integral of PSD(x)*func(x)*dx
            It uses the internal set of diameters to compute the numerical integration
        """
        
        return simps(self(self._diams)*func(self._diams), self._diams)
    
    
    def calc_N(self):
        """calculate numerically the number concentration of the PSD using
        the Simpson's rule and the set of diameters of the PSD
        
        Returns
        -------
        calcN : float
            the actual numerical value of N for this discrete PSD.
            Must always be equal to self.N by construction
        """
        
        return self.integrate(lambda x: 1.0) #simps(self(self._diams), self._diams)
    
    
    def calc_q(self):
        """calculate numerically the mass mixing ratio of the PSD using
        the Simpson's rule and the set of diameters of the PSD
        
        Returns
        -------
        calcq : float
            the actual numerical value of q for this discrete PSD.
            It is used to adjust Lambda and N0 values for optimization
        """
        
        return self.integrate(self.mD) #simps(self(self._diams)*self.mD(self._diams), self._diams)
    
    
    def _optimize_Lambda_N0(self):
        """ Calibrate the values of Lambda and N0 to match self.q and self.N
        Uses bounded optimization formula with bounds set to +- 5% of Lambda
        It is probably best to sort of calculate expected dq/dLambda from
        analytic formula instead
        No return, act on self.Lambda and self.N0
        
        """
        
        def calcPSD(lam):
            pp = PSD.UnnormalizedGammaPSD(self.N0, lam, self.mu, self.maxD)
            pp.N0 *= self.N/simps(pp(self._diams), self._diams)
            return pp
        
        def costFunc(lam):
            pp = calcPSD(lam)
            return np.abs(simps(pp(self._diams)*self.mD(self._diams), self._diams)-self.q)
        
        res = minimize_scalar(costFunc, bounds=[self.Lambda*0.95, self.Lambda*1.05], method='bounded')
        self.Lambda = res.x
        self.N0 *= self.N/self.calc_N()
        
        
        
        