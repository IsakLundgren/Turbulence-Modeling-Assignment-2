import os.path
from matplotlib import ticker
import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from grad_xyz import dphidx,dphidy,dphidz,compute_face,dphidx_2d,dphidy_2d,compute_face_2d,compute_geometry_2d


plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})
plt.interactive(True)

re =9.36e+5 
viscos =1./re

datax= np.loadtxt("x2d_hump_IDDES.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d_hump_IDDES.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])


itstep,nk,dz=np.load('itstep-hump-IDDES.npy')


p2d=np.load('p_averaged-hump-IDDES.npy')/itstep            #mean pressure
u2d=np.load('u_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
v2d=np.load('v_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
w2d=np.load('w_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
k_model2d=np.load('k_averaged-hump-IDDES.npy')/itstep      #mean modeled turbulent kinetic energy velocity
vis2d=np.load('vis_averaged-hump-IDDES.npy')/itstep        #mean modeled total viscosity
uu2d=np.load('uu_stress-hump-IDDES.npy')/itstep
vv2d=np.load('vv_stress-hump-IDDES.npy')/itstep
ww2d=np.load('ww_stress-hump-IDDES.npy')/itstep            #spanwise resolved normal stress
uv2d=np.load('uv_stress-hump-IDDES.npy')/itstep
psi2d=np.load('fk_averaged-hump-IDDES.npy')/itstep         #ratio of RANS to LES lengthscale
eps2d=np.load('eps_averaged-hump-IDDES.npy')/itstep        #mean modeled dissipion of turbulent kinetic energy 
s2_abs2d=np.load('gen_averaged-hump-IDDES.npy')/itstep     #mean |S| (used in Smagorinsky model, the production term in k-eps model, IDDES ...)
s_abs2d=s2_abs2d**0.5

uu2d=uu2d-u2d**2                                #streamwise resolved normal stress
vv2d=vv2d-v2d**2                                #streamwise resolved normal stress
uv2d=uv2d-u2d*v2d                               #streamwise resolved shear stress


kres2d=0.5*(uu2d+vv2d+ww2d)                     


x065_off=np.genfromtxt("x065_off.dat", dtype=None,comments="%")

cyclic_x=False
cyclic_z=True

fx2d,fy2d,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d= compute_geometry_2d(x2d,y2d,xp2d,yp2d)

dudx= np.zeros((ni,nj))
dudy= np.zeros((ni,nj))

u_face_w,u_face_s=compute_face_2d(u2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
v_face_w,v_face_s=compute_face_2d(v2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)

dudy=dphidy_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dvdx=dphidx_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
