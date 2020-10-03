import numpy as np

class Waligorski():
    def __init__(self, t_r,d_step, E_ion, m_ion, eloss, atn1 = 31, atn2 = 7, m1 = 69.723, m2 = 14.0067, density = 6.15):
        atnGa=atn1
        atnN = atn2
        mGa=m1
        mN=m2
        psSgcm3 = density; # Density of GaN


        #nS = psS/((mGa+mN)/2*1.6605402e-27); 
        rhoe = psSgcm3*(atnGa+atnN)/(mGa+mN)

        #rhoe = 2.79

        c = 299792458.0
        me=9.10939e-31
            # [ kg ]   Electron mass

        alpha1=1.667;           # 1.667 for v_ion>0.03
        alpha2=1.079;           # 1.079 for v_ion<0.03

        k = 6.0e-6;   # 6.0e-6 g/cm**2 keV**(-alpha)     

        t = t_r*1e2 * rhoe  #1e2
            #radius in the Waligorski Units (g/cm2)
        m_ion = m_ion*1.6605402e-27;
            # Ion mass in kg  

        E_J_ion = E_ion*1e3*1.602177e-19
            # Ion energy in J
        v_ion=np.sqrt(2*E_J_ion/m_ion);
            # Ion velocity in [m/s]
        v_ion=v_ion/c
            # Ion velocity / c
        if v_ion<0.03:
            alpha=alpha2
        else:
            alpha=alpha1

        Emin= (5.9993+14.5341)/2*1e-3;      #For GaN in KeV
        Emax=2*me*c**2*v_ion**2/(1-v_ion**2)/1.602177e-16  # Maximum transferred energy [ keV ]

        RmaxS=k*Emax**alpha  
           #[g/cm**2] 

        RminS=k*Emin**1.079 
           #[g/cm**2]
            
               # Correction parameters from Waligorski (1986)
        if v_ion>0.03:
            Awal=19.0*v_ion**(1.0/3.0)
        else:
            Awal=8 * v_ion**(1.0/3.0)

        Bwal=0.1e-7;              # [cm], 0.1 nm
        Cwal=1.5e-7+v_ion*5.0e-7;  # [cm] 1.5 nm + 5 nm * beta
        #print(Awal, Bwal, Cwal, sep=' - ')
          
        self.t_r = t_r
        self.t = t
        self.E = E_ion
        self.v_ion = v_ion
        self.eloss = eloss
        self.RmaxS = RmaxS
        self.Rmins = RminS
        self.Emin = Emin
        self.k = k
        self.rhoe = rhoe
        self.d_step = d_step
        self.alpha = alpha
        self.Awal = Awal
        self.Bwal = Bwal
        self.Cwal = Cwal
        
        
        print('\n %.2e MeV'%(E_ion*1e-3))
        print('mass: %f'%m_ion)
        print('beta: %f'%v_ion)
        print('Rmax: %e'%RmaxS)
        print('Rmin: %e'%RminS)
        print('Emin: %f'%Emin)
        print('k: %f'%k)
        print('e density: %0.2f g/cm^3'%rhoe)
        print('e loss: %0.2f keV/nm'%eloss)
        print('d step: %e nm'%(d_step*1e9))
    
    def compute(self):
        t_r = self.t_r
        t = self.t
        E = self.E
        v_ion = self.v_ion
        eloss = self.eloss
        RmaxS = self.RmaxS
        RminS = self.Rmins
        Emin = self.Emin
        k = self.k
        rhoe = self.rhoe
        d_step = self.d_step
        alpha = self.alpha
        Awal = self.Awal
        Bwal = self.Bwal
        Cwal = self.Cwal

        W1D=((1-(t+RminS)/(RmaxS+RminS))**(1/alpha)) /(alpha* v_ion**2 *t*(t+RminS)) 

        mask = t > RmaxS
        W1D[mask] = 0

 

        K1D=Awal*(t-Bwal)/Cwal*np.exp(-(t-Bwal)/Cwal); #Correction factor

        W1D = W1D*(1 + K1D) #[cm2/g]

        total_energy_correct = eloss#*d_step*1e9 #[keV/nm]    
        total_W1D = integrate(t_r*(1e9), W1D) #/ (d_step*1e9)**2 #[nm2 cm2/g]
        normal = total_energy_correct / total_W1D #[keV g/cm2 / nm3]
        #normal = normal #/(d_step*1e9) #[keV/nm g/cm2]
        W1D = W1D * normal            #[keV/nm3]
        
        self.W1D = W1D

        print('Total Energy Deposited: %e' %total_energy_correct)
        print('Normal: %e'%normal)    
        print('W1D sum: %e'%integrate(t_r*(1e9), W1D)) #/(2.79*1e2)
        return W1D
    
    def convert_W2D(self, size, cell_size, n_subcells=11):    
        # [0] also counts so the center is not in 51 but 50 
        t = self.t_r
        W1D = self.W1D

        center_point = (int((size[0])/2), int((size[1])/2))
        W2D = np.zeros(size)
        W2D_subcell = np.zeros((n_subcells, n_subcells))
        
        for ix,iy in np.ndindex(W2D.shape):
            for s_ix, s_iy in np.ndindex(W2D_subcell.shape):
                loc_x = ix + 1/n_subcells*(s_ix-int(n_subcells/2))
                loc_y = iy + 1/n_subcells*(s_iy-int(n_subcells/2))            
                dist_center = np.sqrt((loc_x - center_point[0])**2 + (loc_y - center_point[1])**2)
                dist_center*= cell_size
                    
                if dist_center > t[-1]:
                    energy_cell = 0
                else:
                    energy_cell = np.interp(dist_center, t, W1D)
                    
                W2D_subcell[s_ix, s_iy] = energy_cell
                
            
            W2D[ix, iy] = W2D_subcell.mean()
        
        normal = cell_size * 1e9 * integrate(self.t_r*1e9, W1D)/W2D.sum()  
        W2D = W2D*normal
        self.W2D =W2D

        return W2D
    
def integrate(t, W):
    x = t
    y = W
    sum = 0
    for i in range(len(x)-1):
        sum +=  y[i] * x[i] * (x[i+1] - x[i])
        
    return 2 * np.pi * sum 
