from firedrake import *

grid_size = 100
scale = 10**-6
mesh = IntervalMesh(grid_size,scale)

V = FunctionSpace(mesh, "CG", 1)
Vout = FunctionSpace(mesh, "CG", 1)
W = V * V * V# * V
Wout = Vout * Vout * Vout# *Vout

#a = Function(W, name="Defects")
#a_next = Function(W, name="DefectsNext")
#a_test = TestFunction(W)

#ElectrostaticPotential
#TODO: Do we need a _next function for each??
theta = Function(W)
n, p, v = split(theta)
n_test, p_test, v_test = TestFunctions(W)

#Material boundaries (fractions of the system's lenth)
x_pi = .2*scale
x_in = .8*scale


#COnsts
mu_e = 20 #cm**2/V/s
mu_h = 20 #cm**2/V/s
mu_a = 10**-12 #cm**2/V/s
k_b = 8.617*10**-5 #eV K**-1
q = 1.619*10**-19 #C Electron Charge
gamma_n = 2 # or 1 #fumei
tau_n = 2*10**-15 #s
tau_p = 2*10**-15
Nion_const = 10**19 #/cm**3 Moble Ionic Defect Density
NA_const = 3.0*10**17 #/cm**3 p-type donor density
ND_const = 3.0*10**17 #/cm**3 n-type donor density
N_0 = 10**20 #/cm**3 Effective densitoy of states
T = 300 #K
epsilon_0 = 5.524*10**5 #q**2/eV/cm Vacuum Permittivity
G_default = 2.5*10**21 #cm**3/s Generation rate
k_btb = 10**-10 #cm**3/s Nominal Band-to-band recombination coeff
E_g = 1.6 #eV Band gap

n_i = N_0 * exp(-E_g/(2*k_b*T))


epsilon_r = 20#1#0**-24

#n0 = ?
#p0 = ?
#a0 = ?
x_val = SpatialCoordinate(mesh)#/scale
v0 = x_val[0]/scale

NA = conditional(x_val[0] < x_pi, NA_const, 0)
ND = conditional(x_val[0] > x_in, ND_const, 0)
Nion = conditional(And(x_val[0] > x_pi, x_val[0] < x_in), Nion_const, 0)

n0 = ND_const/10**17#ND
p0 = NA_const/10**17#NA

#it seems G=U=pow(n*p,.5)/(gamma*tau)
#gamma is the recombination reaction order, tau is the SRH recomb. 
#rate coeff for that particular molecule/particle
#I am assuming that Rn=Un here

timestep = 1.0/n

recomb_temp_scaler = 10**-17
G = G_default *recomb_temp_scaler
k1 = k_btb
k2 = n_i**2
U = k1 * (n0*p0 - k2) *recomb_temp_scaler

temp_scaler = k_b/q*T *1.6*10**-16
print(temp_scaler)

#(12)
Ln1 = inner(mu_e*(n*grad(v) - temp_scaler*grad(n)),grad(n_test))
Ln2 = (G - U)*n_test
Ln = (Ln1 + Ln2) * dx
#(13)
Lp1 = inner(mu_h*(-p*grad(v) - temp_scaler*grad(p)),grad(p_test))
Lp2 = (G - U)*p_test
Lp = (Lp1 + Lp2) * dx

temp_scaler02 = 1.6*10**21
#(15)
aV = inner(grad(v),grad(v_test)) * dx
LV1 = 10**17*(n-p)*v_test*dx
LV2 = NA*v_test*dx
LV3 = (-0 + Nion)*v_test*dx
LV4 = -ND*v_test*dx
LVS = (LV1 + LV2 + LV4)
LV = -(q/epsilon_0/epsilon_r) *temp_scaler02* LVS

a_full = aV
L_full = Ln + Lp + LV
res = a_full - L_full
#bcn = DirichletBC(W.sub(0), n0,sub_domain="on_boundary")
#bcp = DirichletBC(W.sub(1), p0,sub_domain="on_boundary")
bcv = DirichletBC(W.sub(2), v0,sub_domain="on_boundary")
#bcn_left
bcn_right = DirichletBC(W.sub(0), n0,sub_domain=2)
bcp_left = DirichletBC(W.sub(1), p0,sub_domain=1)
#bcp_right
#Jn conds how?

#w = Function(W)
#aij single monolythic amtrix
'''solve(res == 0, theta, solver_parameters={'ksp_converged_reason': True,
                                       'ksp_monitor_true_residual': True,
                                       'ksp_view': True
                                         })'''
#quit()

solve(res == 0, theta, bcs=[bcv,bcn_right,bcp_left], 
                              solver_parameters={'mat_type':'aij',
                                          'ksp_type':'preonly',
                                         'pc_type':  'lu',
                                          #'snes_type':'test',
                                         'snes_monitor': True,
                                         'snes_rtol': 10**-30,
                                       #'snes_view': True,
                                       #'ksp_monitor_true_residual': True,
                                       #'snes_converged_reason': True,
                                       'ksp_converged_reason': True
                                       ,'ksp_view': True
                                         })

#File("test.pvd").write(theta)

