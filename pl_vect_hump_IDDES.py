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

datax= np.loadtxt("datnpy/x2d_hump_IDDES.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("datnpy/y2d_hump_IDDES.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])


itstep,nk,dz=np.load('datnpy/itstep-hump-IDDES.npy')


p2d=np.load('datnpy/p_averaged-hump-IDDES.npy')/itstep            #mean pressure
u2d=np.load('datnpy/u_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
v2d=np.load('datnpy/v_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
w2d=np.load('datnpy/w_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
k_model2d=np.load('datnpy/k_averaged-hump-IDDES.npy')/itstep      #mean modeled turbulent kinetic energy velocity
vis2d=np.load('datnpy/vis_averaged-hump-IDDES.npy')/itstep        #mean modeled total viscosity
uu2d=np.load('datnpy/uu_stress-hump-IDDES.npy')/itstep
vv2d=np.load('datnpy/vv_stress-hump-IDDES.npy')/itstep
ww2d=np.load('datnpy/ww_stress-hump-IDDES.npy')/itstep            #spanwise resolved normal stress
uv2d=np.load('datnpy/uv_stress-hump-IDDES.npy')/itstep
psi2d=np.load('datnpy/fk_averaged-hump-IDDES.npy')/itstep         #ratio of RANS to LES lengthscale
eps2d=np.load('datnpy/eps_averaged-hump-IDDES.npy')/itstep        #mean modeled dissipion of turbulent kinetic energy 
s2_abs2d=np.load('datnpy/gen_averaged-hump-IDDES.npy')/itstep     #mean |S| (used in Smagorinsky model, the production term in k-eps model, IDDES ...)
s_abs2d=s2_abs2d**0.5

uu2d=uu2d-u2d**2                                #streamwise resolved normal stress
vv2d=vv2d-v2d**2                                #streamwise resolved normal stress
uv2d=uv2d-u2d*v2d                               #streamwise resolved shear stress


kres2d=0.5*(uu2d+vv2d+ww2d)                     


x065_off=np.genfromtxt("datnpy/x065_off.dat", dtype=None,comments="%")

cyclic_x=False
cyclic_z=True

fx2d,fy2d,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d= compute_geometry_2d(x2d,y2d,xp2d,yp2d)

dudx= np.zeros((ni,nj))
dudy= np.zeros((ni,nj))

u_face_w,u_face_s=compute_face_2d(u2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
v_face_w,v_face_s=compute_face_2d(v2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)

dudy=dphidy_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dvdx=dphidx_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

#Plot psi
fig, ax = plt.subplots()
plt.contourf(xp2d,yp2d,psi2d)
ax.set_title("$\\psi$ in the domain")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
plt.colorbar()
plt.savefig("img/psi.eps")

#Calculate wall distance
calcWallDist = False

if calcWallDist:
    dw = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            Distances = np.zeros((ni,2))
            for k in range(ni):
                Distances[k,0] = np.sqrt((xp2d[i,j]-xp2d[k,1])**2 + (yp2d[i,j]-yp2d[k,1])**2)
                Distances[k,1] = np.sqrt((xp2d[i,j]-xp2d[k,nj-1])**2 + (yp2d[i,j]-yp2d[k,nj-1])**2)
            dw[i,j] = np.min(Distances)
            np.disp("Cell " + str(i * nj + j) + " out of " + str(ni * nj - 1) + " done!")
    
    # Write Walldistance vector to file
    np.savetxt("datnpy/Walldistance.dat", dw)
else:
    if os.path.isfile("datnpy/Walldistance.dat"):
        # Read Walldistance vector from file
        dw = np.loadtxt("datnpy/Walldistance.dat")
    else:
        print("Error: Walldistance file not found.")

#Calculate hmax
calchmax = False
if calchmax:
    hmax = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            sidesize = np.zeros(4)
            sidesize[0] = (x2d[i,j+1]-x2d[i,j])**2 + (y2d[i,j+1]-y2d[i,j])**2
            sidesize[1] = (x2d[i+1,j+1]-x2d[i,j+1])**2 + (y2d[i+1,j+1]-y2d[i,j+1])**2
            sidesize[2] = (x2d[i+1,j]-x2d[i+1,j+1])**2 + (y2d[i+1,j]-y2d[i+1,j+1])**2
            sidesize[3] = (x2d[i,j]-x2d[i+1,j])**2 + (y2d[i,j]-y2d[i+1,j])**2
            hmax[i,j] = np.sqrt(np.max(sidesize))
            np.disp("Cell " + str(i * nj + j) + " out of " + str(ni * nj - 1) + " done!")
    
    # Write Walldistance vector to file
    np.savetxt("datnpy/hmax.dat", hmax)
else:
    if os.path.isfile("datnpy/hmax.dat"):
        # Read Walldistance vector from file
        hmax = np.loadtxt("datnpy/hmax.dat")
    else:
        print("Error: hmax file not found.")


#Plot f_d tilde TODO Go back and review this, wrong somewhere
kappa2 = 0.41**2
vis_t2d = vis2d-viscos
denominator = kappa2 * np.multiply(dw**2, np.maximum(s_abs2d,10**(-10)))
r_dt = np.divide(vis_t2d,denominator)
alpha = 0.25 - np.divide(dw,hmax)
fdt = 1 - np.tanh((8 * r_dt)**3)
fB = np.minimum(2 * np.exp(-9*alpha),1)
fdtild = np.maximum(1-fdt,fB)
stations = [0.66, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\\tilde{f_d}$ for different $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-xp2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(fdtild[iinner,:],yp2d[iinner,:],'b-')
    ax[i].set_title("$x$ = " + str(stations[i]))
fig.savefig('img/dftilde.eps')

#Plot f_dt
fig, ax = plt.subplots()
plt.contourf(xp2d,yp2d,fdt)
ax.set_title("$f_{dt}$ in the domain")
ax.set_ylabel("$y$")
ax.set_xlabel("$x$")
plt.colorbar()
plt.savefig("img/fdt.eps")

plt.show(block=True)