from firedrake import *

grid_size = 100
scale = 10**-6
mesh = IntervalMesh(grid_size,scale)

V = FunctionSpace(mesh, "CG", 1)
Vout = FunctionSpace(mesh, "CG", 1)

v = Function(V)
v_test = TestFunction(V)

#Material boundaries (fractions of the system's lenth)
x_pi = .2*scale
x_in = .8*scale

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
T = 300 #K
epsilon_0 = 5.524*10**5 #q**2/eV/cm Vacuum Permittivity
G_default = 2.5*10**21 #cm**3/s
epsilon_r = 20#1#0**-24

x_val = SpatialCoordinate(mesh)#/scale
v0 = 1-x_val[0]/scale

NA = conditional(x_val[0] < x_pi, NA_const, 0)
ND = conditional(x_val[0] > x_in, ND_const, 0)
Nion = conditional(And(x_val[0] > x_pi, x_val[0] < x_in), Nion_const, 0)

temp_const02 = 1.6*10**21
#LV1 = 10**17*(n-p)*v_test*dx
LV2 = NA*v_test*dx
LV3 = 10**-3*(-0 + Nion)*v_test*dx
LV4 = -ND*v_test*dx
LVS = (LV2 + LV4)
L = (q/epsilon_0/epsilon_r *temp_const02)* LVS

#L = 2.*v_test*dx
a = inner(grad(v),grad(v_test)) * dx
res = a - L

bcv = DirichletBC(V, v0,sub_domain="on_boundary")

solve(res == 0, v, bcs=[bcv], 
                              solver_parameters={'mat_type':'aij',
                                          'ksp_type':'preonly',
                                         'pc_type':  'lu',
                                          #'snes_type':'test',
                                         'snes_monitor': True,
                                       #'snes_view': True,
                                       #'ksp_monitor_true_residual': True,
                                       #'snes_converged_reason': True,
                                       'ksp_converged_reason': True
                                       ,'ksp_view': True
                                         })

