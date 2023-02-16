import numpy
import pylab as plt
from numpy.random import rand, randn, seed
from pylab import log10, sqrt, log, exp, pi
from scipy.special import expn, expi, hyp2f1
from scipy.integrate import ode, quad, simps
from scipy import interpolate
from numpy import random

from scipy.optimize import fsolve
from os.path import exists
import pickle
from numpy import shape

# Routine to the evolve the GCMF at different Galactcocentric radii
class EvGcmf:
    def __init__(self, **kwargs):
        r""" Evolve GCMF(R) """
        self._set_kwargs(**kwargs)
        self.evolve()
        self.get_moments()
        self._scale()
        self.output()
        
    # Functions
    def _set_kwargs(self, **kwargs):
        # Default settings are for Model VII in Gieles & Gnedin
        
        # Evolution
        self.age = 12e3 # [Myr]
        self.x = 2/3
        self.y = 4/3
        self.Mdotref = 44.4 # [Msun/Myr]
        
        # Mass function
        self.Mlo = 1e2 # [Msun]
        self.Mup = 1e8 # [Msun]
        self.Mc = 8e5 # [Msun] Schechter mass
        self.Mref = 2e5 # [Msun]
        self.Mlim = numpy.array([1e2, 1e4, 0.55e5]) # [Msun] Minimum masses for mass loss in field calc

        # Density
        self.gamma = 3.3
        self.delta = 0.5
        
        # Galaxy
        self.Rlo = 1 # [kpc]
        self.Rup = 100 # [kpc]
        self.Vc = 220 # [km/s]
        self.Rani = 5 # [kpc] Anisotropy radius

        # Additional params
        self.feh_grad = True
        self.past_evo = True

        # Numerical constants defining resolution
        self.NR = 30 # number of R points
        self.NM = 50 # number of MF points
        self.NV = 25 # number of points on velocity grid
        self.ns = 5  # number of sigma  
        self.ft = 1/200 # fraction of vt for integration boundaries

        self.NGC = 156 # Final number of GCs the model is scaled to
        if kwargs is not None:
            for key, value in kwargs.items(): 
                setattr(self, key, value)

                
        self.Mdotref = abs(self.Mdotref) # ensure positive
                
    def icmf(self,M,index):
        # Initial cluster mass function

        # index = 0 => N = 1
        # index = 1 => <M>
        # index = 2 => <M^2> etc
        A = (exp(-self.Mlo/self.Mc)/self.Mlo + expi(-self.Mlo/self.Mc)/self.Mc)**(-1)
        return A*M**index*M**-2*exp(-M/self.Mc)

    def get_nR0(self,R):
        delta, gamma = self.delta, self.gamma
        Rlo, Rup, Ra = self.Rlo, self.Rup, self.Rani
        a = 2/delta
        b = (-1+gamma)/delta
        c = (-1+delta+gamma)/delta
        
        g1 = 1-gamma
        Pu = hyp2f1(a,b,c,-(Ra/Rup)**delta)*(Rup/Rlo)**g1
        Pl = hyp2f1(a,b,c,-(Ra/Rlo)**delta)
        A = g1/(4*pi*Ra**2*Rlo**g1*(Pu - Pl))

        return A*R**-gamma/(1+(R/Ra)**delta)**(2./delta)

    def beta_func(self, R):
        return 1/(1+(self.Rani/R)**self.delta) 

    def get_y_Mdotref(self,R):
        if not hasattr(R,"__len__"): R = numpy.array([R])
        logR = log10(R)

        feh = numpy.zeros_like(R) - 1.5
        c = (R<10)
        feh[c] = -0.5-logR[c]

        y = numpy.zeros_like(R) + 4./3
        Mdot = numpy.zeros_like(R) + self.Mdotref

        Mdot[c] *= 2./3 + 1./3*((logR[c]+0.5)/1.5) 
        y[c] = 2./3 + 2./3*((logR[c]+0.5)/1.5) 
        
        return y, Mdot, feh

    # Evolve main function
    def evolve(self):
        NR, NM, NV = self.NR, self.NM, self.NV
        
        lRedge = numpy.linspace(log10(self.Rlo), log10(self.Rup), NR+1)
        lR = 0.5*(lRedge[1:] + lRedge[0:-1])
        R = 10**lR
        dR = 10**lRedge[1:] - 10**lRedge[0:-1]
        
        # weights for each radial bin
        w = 4*pi*R**2*self.get_nR0(R)*dR

        mVr2 = numpy.zeros(NR)
        mVt2 = numpy.zeros(NR)
        self.beta = numpy.zeros(NR)
        
        dNdM = numpy.zeros((NR, NM))
        Mdis = numpy.zeros((NR, len(self.Mlim)))
        beta = numpy.zeros((NR))

        # Setup interpolator for orbits, once
        self.orbit_interp = self.get_orbit_interpolator()

        for i in range(NR):
            
            # Setup velocity grid
            sigr = self.get_sigr(R[i])
            sigt = sigr*sqrt(2*(1-self.beta_func(R[i])))

            Vr = numpy.linspace(-self.ns*sigr,self.ns*sigr, NV)
            Vt = numpy.linspace(self.ft*sigt,self.ns*sigt, NV)
            dVr = Vr[1]-Vr[0]
            dVt = Vt[1]-Vt[0]
            Vrg, Vtg = numpy.meshgrid(Vr, Vt)

            # Compute d^2N/(dVrdVt)
            rhoVg = self.get_rhoV_grid(Vrg, Vtg, sigr, sigt)

            # Get Rc, eta and orbital parameters
            Rcg, etag  = self.get_Rc_eta_grid(R[i], Vrg, Vtg)
            eccg, Rpcg, Racg = self.orbit_interp(etag)
            Rpg = Rpcg*Rcg
            Rag = Racg*Rcg
            Reg = Rpg*(1+eccg)
        
            # Compute MF on grid
            M0g, Mg, dNdMg, Mdisg = self.get_dNdM_grid(Reg,R[i])

            #print(" int F(Vr, Vt) = ",simps([simps(zz_vr, Vr) for zz_vr in rhoVg],Vt))
            
            # Now integrate MF and mass loss
            for j in range(NM):
                zz = rhoVg*dNdMg[:,:,j]
                dNdM[i][j] = simps([simps(zz_vr, Vr) for zz_vr in zz],Vt)

            for j in range(len(self.Mlim)):
                zz = rhoVg*Mdisg[:,:,j]
                Mdis[i][j] = simps([simps(zz_vr, Vr) for zz_vr in zz],Vt)

            # Get anisotropy of massive clusters
            N5surv = 0
            rho = 0
            for j in range(NV):
                for k in range(NV):
                    c = (Mg[j,k,:]>1e5)
                    N5 = simps(dNdMg[j,k,c],x=Mg[j,k,c])
                    N5surv += N5
                    mVr2[i] += rhoVg[j,k]*Vrg[j,k]**2*N5*dVr*dVt
                    mVt2[i] += rhoVg[j,k]*Vtg[j,k]**2*N5*dVr*dVt
                    rho += rhoVg[j,k]*dVr*dVt*N5
            mVr2[i]/=rho
            mVt2[i]/=rho
#            print(" Vr ",i,sqrt(mVr2[i]),sqrt(mVt2[i]),sigr, rho)
        #    self.beta[i] = 1-mVt2[i]/(2*mVr2[i])
        self.beta = 1-mVt2/(2*mVr2)
        self.Vr2 = mVr2
        self.Vt2 = mVt2
        
        nR0, nR, Nsurv, Msurv = self.get_nR(R, Mg[0,0], dNdM, w, dR)
        rhodis, Mdis = self.get_rhodis(R, Mdis, w, dR)

        self.R = R
        self.nR0 = nR0
        self.nR = nR
        self.w = w
        self.Ni = 1
        self.Mi = quad(self.icmf, self.Mlo, self.Mup,args=(1))[0]
        self.Nsurv = Nsurv
        self.Msurv = Msurv
        self.rhodis = rhodis
        self.Mdis = Mdis

        self.M = Mg[0,0]
        self.dNdM = dNdM

        self.dNdMtot = numpy.zeros(NM)
        for i in range(NR):
            self.dNdMtot += dNdM[i]*w[i]

        return  

    def get_orbit_interpolator(self):
        # Compute eta(ecc), solve roots for Ra/Rc and setup interpolator
        
        def jac_func(x,eta):
            return numpy.array([-2/x**3 + 2/(eta**2*x)])

        def orbit_func(x, eta):
            return 1/x**2 + 2/eta**2 *log(x) - 1/eta**2

        N=100
        ecc = numpy.linspace(1,0,N)

        # Take care with ecc=0 and ecc=1
        rp = (1-ecc[1:-1])/(1+ecc[1:-1])
        vp = sqrt(-2*log(rp)/(1-rp**2))
        eta = numpy.r_[0,vp/exp(0.5*(vp**2-1)), 1]

        Rac = numpy.zeros(N)
        for i in range(1,N-1):
            Rac[i] = fsolve(orbit_func, exp(0.5),args=(eta[i]), fprime=jac_func)

        # Take care with the extremes
        Rac[0], Rac[-1]= exp(0.5), 1
        Rpc = Rac*(1-ecc)/(1+ecc)
        f = interpolate.interp1d(eta, numpy.array([ecc, Rpc, Rac]))
        return f

    def get_sigr(self, R):
        sigr2 = self.Vc**2/self.gamma    
        return sqrt(sigr2)

    def get_rhoV_grid(self, Vr, Vt, sr, st):
        sr2, st2 = sr**2, st**2
        fr = exp(-Vr**2/(2*sr2))/sqrt(2*pi*sr2)
        ft = exp(-Vt**2/st2)/(pi*st2)
        return 2*pi*Vt* fr*ft
    
    def get_Rc_eta_grid(self,R, Vr, Vt):
        # Get Peri and Apo
        J = R*Vt
        V2 = Vr**2 + Vt**2
        E = 0.5*V2 + self.Vc**2*log(R)
        
        Rc = exp( (E-self.Vc**2/2)/self.Vc**2)
        eta = J/(Rc*self.Vc)
        
        return Rc, eta

    def get_dNdM_grid(self,Re,R):
        # Compute mass function on grid
        x,y,Mref,age = self.x, self.y, self.Mref, self.age
        Mdotref = self.Mdotref

        if (self.feh_grad):
            y, Mdotref, FeH = self.get_y_Mdotref(R)

        Mlo = self.Mlo
        NV, NR = self.NV, self.NR
        Mlim = self.Mlim
        
        # Populations masses needed for mass loss
        Mpop = numpy.zeros_like(Mlim)
        for i in range(len(Mlim)):
            Mpop[i] = quad(self.icmf,Mlim[i], self.Mup, args=(1))[0]
        self.Mpop = Mpop
        
        # Mdot grid
        Mdot = Mdotref/Re
            
        if (self.past_evo):
            c = (Re>4)
            Mdot[c]*=(Re[c]/4)**0.5

        M = numpy.logspace(log10(self.Mlo),log10(self.Mup),self.NM)

        Mdis = numpy.stack([numpy.zeros_like(Mdot) for _ in range(len(Mlim))], axis=2)
        
        # Generate M0 only for surviving masses
        Mmin = Mref*(y*age*Mdot/Mref)**(1./x) # NVxNV
        Mmin[(Mmin<Mlo)] = Mlo

        Mlog = numpy.zeros_like(Mmin) + Mlo
#        print(" minmax = ",Mmin.min(), Mmin.max())

        # Now 3d grid : NVxNVxNM
        Mp = numpy.logspace(log10(Mlog),log10(self.Mup),self.NM).T 

        # can be improved by form: M0 = Mmin + logspace(Mlo, Mup) ....
#        M0 = numpy.logspace(log10(Mmin),log10(self.Mup),self.NM).transpose(1,2,0)

        M0 = numpy.logspace(log10(Mlog),log10(self.Mup),self.NM) + Mmin
        M0 = M0.transpose(1,2,0)
        
        Mdot3 = numpy.stack([Mdot for _ in range(self.NM)], axis=2)
        
        M = numpy.zeros_like(M0)
        tdis = (1./y)*Mref/Mdot3*(M0/Mref)**x
    
        D = age/tdis
        c = (D<1)
        
        M[c] = M0[c]*(1 - D[c])**(1./y)
        M0p = numpy.zeros_like(M0)

        # Can be pythonized?
        for i in range(NV):
            for j in range(NV):
                M0p[i,j] = numpy.interp(Mp[i,j], M[i,j], M0[i,j])

        tdisp = (1./y)*Mref/Mdot3*(M0p/Mref)**x
        Dp = age/tdisp
        cp = (Dp < 1)
        
        dMdM0p = numpy.zeros_like(M0p)
        dNdMp = numpy.zeros_like(M0p)
    
        if numpy.count_nonzero(cp)>0:
            dMdM0p[cp] = (1 - Dp[cp])**(1./y) + Dp[cp]*(1-Dp[cp])**(1/y-1)*x/y
            dNdMp[cp] = self.icmf(M0p[cp],0)/dMdM0p[cp]

            # Can be pythonized?
            for i in range(NV):
                for j in range(NV):
                    for k in range(len(Mlim)):
                        c = (M0p[i,j,:]>=Mlim[k])
                        Mdis[i,j,k] = Mpop[k] - simps(dNdMp[i,j,c]*Mp[i,j,c],x=Mp[i,j,c])
        else:
            print(" NOT")

        return M0p, Mp, dNdMp, Mdis

    def get_nR(self,R, M, dNdM, w, dR):
        nR = numpy.zeros(self.NR)        
        nR0 = numpy.zeros(self.NR)
        Nsurv = 0
        Msurv = 0
        Ntot = 0
        for i in range(self.NR):
            Nc = simps(dNdM[i], x=M)
            Mc = simps(dNdM[i]*M, x=M)        
            nR0[i] = w[i]/(4*pi*R[i]**2*dR[i])
            nR[i] =  Nc*w[i]/(4*pi*R[i]**2*dR[i])
            Nsurv += Nc*w[i]
            Msurv += Mc*w[i]
            Ntot += w[i]

        return nR0, nR, Nsurv, Msurv

    def get_rhodis(self,R, Mdis, w, dR):
        # mass density profile of escapers
        rhodis = numpy.zeros((self.NR,len(self.Mlim)))

        for i in range(self.NR):
            fac = w[i]/(4*pi*R[i]**2*dR[i])
            rhodis[i,:] = fac*Mdis[i,:]
            Mdis[i,:] *= w[i]
            
        return rhodis, Mdis

    def get_moments(self):
        mlogM = numpy.zeros(self.NR)
        slogM = numpy.zeros(self.NR)
        X = log10(self.M)
            
        for i in range(self.NR):
            Y = self.M*self.dNdM[i]
            area = simps(Y,x=X)
            mlogM[i] = simps(X*Y,x=X)/area
            slogM[i] = sqrt(simps((X-mlogM[i])**2*Y,x=X)/area)
        self.mlogM = mlogM
        self.slogM = slogM

    def _scale(self):
        # Scale the number of surviving to th number of MW GCs
        self.scale_fac = self.NGC/self.Nsurv

        self.nR0 *= self.scale_fac
        self.nR *= self.scale_fac
        self.Nsurv *= self.scale_fac
        self.Msurv *= self.scale_fac

        self.Ni *= self.scale_fac
        self.Mi *= self.scale_fac
        self.rhodis *= self.scale_fac
        self.Mdis *= self.scale_fac
        self.dNdM *= self.scale_fac
        self.dNdMtot *= self.scale_fac
        self.dNdM0 = self.icmf(self.M,0)*self.scale_fac
        
    def output(self):
        print("\n Parameters: age = %4.2f Gyr; Mdotref = %4.1f Msun/Myr; x = %4.2f; y = %4.2f"%(self.age/1e3, -self.Mdotref, self.x, self.y))
        print("             Mc = %6.2e Msun; Mlo = %6.2e Msun; Mup = %6.2e Msun"%(self.Mc, self.Mlo, self.Mup))
        print("             gamma = %3.1f; Rani = %3.1f kpc; delta = %4.2e"%(self.gamma, self.Rani, self.delta))
        print("             NM = %3i; NR = %3i; NV = %3i"%(self.NM, self.NR, self.NV))
        print(" Initial number  = %7.2e"%self.Ni)
        print(" Initial mass    = %7.2e Msun"%self.Mi)
        print(" Final number    = %7.2e"%self.Nsurv)
        print(" Final mass      = %7.2e Msun"%self.Msurv)
        print(" Average GC mass = %7.2e Msun"%(self.Msurv/self.Nsurv))

        DM = sum(self.Mdis[:,0])
        print(" Mass lost       = %7.2e %7.2e Msun"%(self.Mi-self.Msurv,DM))


    # All functions below are for Monte Carlo sampling
    def sample(self,N):
        print(" sample N = ",N)
        self.N = int(N)
        self.initialise()
        self.evolve_sample(self.M0s)

    def sample_R(self):
        Rlo, Rup, Ra, delta , gamma= self.Rlo, self.Rup, self.Rani, self.delta, self.gamma
        a = 2/delta
        b = (-1+gamma)/delta
        c = (-1+delta+gamma)/delta
        
        NR = 100
        R = numpy.logspace(log10(Rlo), log10(Rup), NR)
        Plo = hyp2f1(a,b,c,-(Ra/Rlo)**delta)
        C = Plo  - hyp2f1(a,b,c,-(Ra/Rup)**delta)*(Rup/Rlo)**(1-gamma)
        CDF = (Plo - hyp2f1(a,b,c,-(Ra/R)**delta)*(R/Rlo)**(1-gamma))/C
        RAN = rand(self.N)
        return numpy.interp(RAN, CDF, R)

    def sample_icmf(self):
        Ni = 1000

        # Note: sample from Mlo = 10^4
        Mlo, Mup,Mc = 1e4, self.Mup, self.Mc
        X = numpy.logspace(log10(Mlo),log10(Mup),Ni)
        a1 = (exp(-Mlo/Mc)/Mlo + expi(-Mlo/Mc)/Mc)
        a2 = (exp(-X/Mc)/X + expi(-X/Mc)/Mc)
        CDF = (a1 - a2)/a1
        RAN = rand(self.N)
        return numpy.interp(RAN, CDF, X)

    def sample_velocities(self,R):
        sigr = self.Vc/sqrt(self.gamma) 
        Vr = randn(self.N)*sigr
        Vtheta = randn(self.N)*sigr*sqrt(1-self.beta_func(R))
        Vphi = randn(self.N)*sigr*sqrt(1-self.beta_func(R))
        Vt = sqrt(Vtheta**2 + Vphi**2)
        return Vr, Vt

    def initialise(self):
        self.Rs = self.sample_R()
        self.M0s = self.sample_icmf()
        self.Vrs, self.Vts = self.sample_velocities(self.Rs)

    def evolve_sample(self, M0):
        Rc, eta = self.get_Rc_eta_grid(self.Rs, self.Vrs, self.Vts)
        ecc, Rpc, Rac = self.orbit_interp(eta)
        
        Rp = Rpc*Rc
        Ra = Rac*Rc
        Re = Rp*(1+ecc)

        self.Res = Re

        M = numpy.zeros_like(M0)

        x, y, Mref, Mdotref = self.x, self.y, self.Mref, self.Mdotref
        y = numpy.zeros_like(M) + y
        
        if (self.feh_grad):
            y, Mdotref, FeH = self.get_y_Mdotref(self.Rs)

        if (self.past_evo):
            c = (Re>=4)
            Re[c]/=sqrt(Re/4)

        Mdot = Mdotref/Re

#        Mmin = min(self.Mref*(y*age*Mdot/self.Mref)**(1./self.x))

        self.rhoh0s = 300*(Re/9.2)**-2
                
        tdis = (1./y)*Mref/Mdot*(M0/Mref)**x
        D = self.age/tdis
        c = (D<1)
        
        M[c] = M0[c]*(1 - D[c])**(1./y[c])
        self.Ms = M
        clo = (M>self.Mlo)
        self.Ns = numpy.count_nonzero(clo)

