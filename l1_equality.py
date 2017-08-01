"""L1 minimization with equality constraints.
	min L1_norm(x)
	s.t. Ax = b
"""

import numpy as np
import copy
import numbers
from cs.optimize import PrimalDualForLP
from cs.optimize import BaseNewton


class L1EqualitySolver(BaseNewton):
    """L1 minimization with equality constraints using primal-dual interior point method.
  
    Parameters:
    -----------
    A: ndarray, list or tuple [m,n] 
        Linear system Ax = b
        
    b: ndarray, list or tuple [m,1] 
        Linear system Ax = b
    
    initial_solution: ndarray, list or tuple optional default []
        Initial point
    
    tolerance: float optional default 1e-3
        If the residual falls below this tolerance, iteration will be terminated.
    max_iters: int optional default 100
        Maximum iterations allowed
    enable_staging: bool optional default False
        If True, some intermediate information are stored and can be displayed or 
        plotted later on.
    
    verbose: bool optional default False
        Show some intermediate information, if True. Silence if False.
    """
    def __init__(self, 
                  A, 
                  b,
                  initial_solution=[],
                  tolerance=1e-3,
                  max_iters=50,
                  enable_staging=False,
                  verbose=False):
        
        super(L1EqualitySolver, self).__init__(initial_solution=initial_solution,
                                                tolerance=tolerance,
                                                max_iters=max_iters,
                                                enable_staging=enable_staging,
                                                verbose=verbose)
        
        self.A=np.array(A)
        self.b=np.array(b)         
        self.tau = 2.0
        self.tau_base = 2.0
 
        self.n = self.A.shape[1] # dimension of x
        self.m = self.A.shape[0] # Number of equality constraints
 
        #
        # Solution format: 
        #   primal variables: x u 
        #   dual variables:   v lamda_u1, lamda_u2
        #   x, u, v, lamda_u1, lamda_u2
        #
        self.x = None
        self.u = None
        self.v = None
        self.lamda_u1 = None
        self.lamda_u2 = None
        
        self.dx = None
        self.du = None
        self.dv = None
        self.dlamda_u1 = None
        self.dlamda_u2 = None
        
        self.step_size = 1.0        
        self.cond_number = 0.0
        self.sdg = 1e10
        
        self.staged_residual = []
        self.staged_step_size = []
        self.staged_sdg = []        
        self.staged_cond_number = []
        

    def _check_stop_criteria(self):
        """Check whether stop criteria are satisfied. OVERRIDDEN"""
        sdg_below_tole =self.sdg < self.tolerance 
        if sdg_below_tole:
            print "Surrogate duality gap is less than the tolerance ", self.tolerance
        return sdg_below_tole or  \
               super(L1EqualitySolver, self)._check_stop_criteria()


    def _initialize(self):
        """Get an initial solution"""
        if self.initial_solution != []:
            return self.initial_solution        
        
        #
        # Use the Minimum-Energy Solution as the iniitial solution
        #
        AAt_inv = np.linalg.inv(np.dot(self.A, self.A.transpose()))
        x0 = np.dot(np.dot(self.A.transpose(),AAt_inv),b)

        self.x = x0
        self.u = 0.95*np.abs(x0) + 0.10*np.max(np.abs(x0));          # Bounds        
        self.lamda_u1 = -1.0/(self.x-self.u)   # Dual variables for inequality constraints
        self.lamda_u2 = -1.0/(-self.x-self.u)  # Dual variables for inequality constraints        
        self.v = -np.dot(self.A, self.lamda_u1-self.lamda_u2)        # Dual variables for 
                                                                     # equality constraints
        #self.lamda_u1 = np.ones((self.n,),dtype=np.float64)
        #self.u = np.ones((self.n,),dtype=np.float64)
        #self.v = np.ones((self.m,),dtype=np.float64)
        self.dx = 0.0
        self.du = 0.0
        self.dv = 0.0
        self.dlamda_u1 = 0.0
        self.dlamda_u2 = 0.0
            

    def _newton_step(self):
        """Compute a feasible Newton step.
        
        Warning:
        --------
        As inversion is involved in the method, to avoid divide-by-zero, elements that are zeros are
        replaced by a small positive constant 1e-20. Be aware and make sure that this won't affect 
        the precision.
        """
        x, u, v, lamda_u1, lamda_u2 = self.x, self.u, self.v, self.lamda_u1, self.lamda_u2        
        epsilon = 1e-10 # Avoid dividing-by-zero
                
        fu1 = x-u 
        fu2 = -x-u        
        fu1[np.where(fu1==0)] = epsilon
        fu2[np.where(fu2==0)] = epsilon
        fu1_inv = 1.0/fu1
        fu2_inv = 1.0/fu2
        
        Sigma1 = - lamda_u1*fu1_inv - lamda_u2*fu2_inv
        Sigma1[np.where(Sigma1==0)] = epsilon
        Sigma1_inv = 1.0/Sigma1
        
        Sigma2 = lamda_u1*fu1_inv - lamda_u2*fu2_inv
        
        Sigmax = Sigma1 - np.power(Sigma2,2.0)*Sigma1_inv
        Sigmax[np.where(Sigmax==0)] = epsilon
        Sigmax_inv = 1.0/Sigmax
       
        #
        # Complementary slackness
        #
        self.sdg = - (np.dot(x-u, lamda_u1) + np.dot(-x-u, lamda_u2))
        self.tau = self.tau_base*2*self.n/self.sdg
        #self.tau *= self.tau_base
        
        w1 = -1.0/self.tau * (-fu1_inv + fu2_inv) - np.dot(self.A.transpose(),v)
        w2 = -1 - (1.0/self.tau)*(fu1_inv + fu2_inv)
        w3 = b - np.dot(A, x)                   
        
        #
        # H * dv = d
        #
        S = np.eye(self.n,dtype=np.float64)       
        np.fill_diagonal(S, Sigmax_inv)
               
        H = - np.dot(np.dot(self.A, S), 
                     self.A.transpose())
        self.cond_number = np.linalg.cond(H)
        self.staged_cond_number.append(self.cond_number)
        
        bv = w3-np.dot(A, Sigmax_inv*w1-Sigmax_inv*Sigma2*Sigma1_inv*w2)
        dv = np.dot(np.linalg.inv(H),bv)

        dx = Sigmax_inv*(w1-Sigma2*Sigma1_inv*w2-np.dot(self.A.transpose(),dv))
        du = Sigma1_inv*(w2-Sigma2*dx)
        
        dlamda_u1 = lamda_u1*fu1_inv*(-dx+du)-lamda_u1-(1.0/self.tau)*fu1_inv
        dlamda_u2 = lamda_u2*fu2_inv*(dx+du)-lamda_u2-(1.0/self.tau)*fu2_inv 
        
        self.dx, self.du, self.dv, self.dlamda_u1, self.dlamda_u2 = \
            dx, du, dv, dlamda_u1, dlamda_u2       
        
        return dx, du, dv, dlamda_u1, dlamda_u2  
               
        
    def _step_size(self):
        """Determine a good step size OVERRIDDEN"""
        x,u,v,lamda_u1,lamda_u2 = self.x, self.u, self.v, self.lamda_u1, self.lamda_u2
        dx,du,dv,dlamda_u1,dlamda_u2 = self.dx, self.du, self.dv, self.dlamda_u1, self.dlamda_u2

        #
        # First make sure that the step is feasible: keep lamda_u1, lamda_u1 > 0, fu1, fu2 <0
        #
        Ild_u1 = np.where(dlamda_u1<0)
        Ild_u2 = np.where(dlamda_u2<0)
        s = min([1, np.min(-lamda_u1[Ild_u1]/dlamda_u1[Ild_u1]),
                    np.min(-lamda_u2[Ild_u2]/dlamda_u2[Ild_u2])])
        
        Ifu1 = np.where((dx-du)>0)
        Ifu2 = np.where((-dx-du)>0)        
        s = 0.99*min([s, np.min((-(x-u)[Ifu1]/(dx-du)[Ifu1])),
                         np.min((x+u)[Ifu2]/(-dx-du)[Ifu2])])        
        
        #
        # Backtracking line search
        #
        good = False
        max_iters = 32
        alpha = 0.01
        beta = 0.5
        count = 0
        rc0,rd0,rp0,rn0 = self._residual(solution=[x, u, v, lamda_u1, lamda_u2])
        while count < max_iters:            
            rc,rd,rp,rn = self._residual(solution=[x+s*dx, 
                                                   u+s*du, 
                                                   v+s*dv,     
                                                   lamda_u1+s*dlamda_u1, 
                                                   lamda_u2+s*dlamda_u2])
                    
            if rn <= (1.0-s*alpha)*rn0:
                good = True
                break
            s *= beta
            count += 1            
        self.step_size = s
        return good
    
    
    def _residual(self, solution):
        """Calcuate residuals OVERRIDDEN"""
        [x, u, v, lamda_u1, lamda_u2] = solution
        fu1 = x-u
        fu2 = -x-u
        r_cent=np.hstack((-lamda_u1*fu1,-lamda_u2*fu2))-1.0/self.tau
        r_dual = np.hstack(((lamda_u1-lamda_u2) + np.dot(self.A.transpose(),v),
                            1-lamda_u1-lamda_u2))        
        r_pri = np.dot(self.A,x) - self.b  
        r_norm = np.sqrt(np.sum(r_dual*r_dual) + np.sum(r_pri*r_pri) + np.sum(r_cent*r_cent))
        return r_cent, r_dual, r_pri, r_norm
        
    
    def _update_solution(self):
        """Update a solution"""        
        self.x += self.dx * self.step_size
        self.u += self.du * self.step_size
        self.v += self.dv * self.step_size
        self.lamda_u1 += self.dlamda_u1 * self.step_size
        self.lamda_u2 += self.dlamda_u2 * self.step_size
                
        
    def _staging(self):
        """Store intermediate infos"""
        rc,rd,rp,rn = self._residual(solution=[self.x, 
                                               self.u, 
                                               self.v, 
                                               self.lamda_u1, 
                                               self.lamda_u2])        
        self.staged_residual.append(rn)
        self.staged_step_size.append(copy.copy(self.step_size))
        self.staged_sdg.append(self.sdg) 
        
        if self.verbose:
            print 'Iteration: ', self.iter, \
                  ' Cond num: ',   "{:.3e}".format(self.cond_number), \
                  ' SDG: ',        "{:.3e}".format(self.sdg), \
                  ' Primal res: ', "{:.3e}".format(np.sqrt(np.dot(rp,rp))), \
                  ' Dual res: ',   "{:.3e}".format(np.sqrt(np.dot(rd,rd))), \
                  ' Cent res: ',   "{:.3e}".format(np.sqrt(np.dot(rc,rc))), \
                  ' Tau: ',        "{:e}".format(self.tau)                  
                           
                              
    def solve(self):
        """Solve a l1-min with equality constraints"""                
        self._iterate()
        return self.x
