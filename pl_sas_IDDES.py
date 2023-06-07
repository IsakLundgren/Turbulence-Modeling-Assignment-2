import os.path
import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
from grad_xyz import dphidx_2d,dphidy_2d,compute_face_2d,compute_geometry_2d, \
                     dphidx,dphidy,dphidz,compute_face,compute_geometry
from astropy.stats import circcorrcoef
plt.rcParams.update({'font.size': 22})
plt.interactive(True)
re =9.36e+5
viscos =1/re

datax= np.loadtxt('datnpy/x2d_hump_IDDES.dat')
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt('datnpy/y2d_hump_IDDES.dat')
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])


# z grid
zmax, nk=np.loadtxt('datnpy/z_hump_IDDES.dat')
nk=int(nk)
dz=zmax/nk
zp2d = np.linspace(0,zmax, num=nk)

# loop over nfiles 
#nfiles=23
nfiles=2
#initialize fields
u3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
v3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
w3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))

for n in range(0,(nfiles)):
   nn=n*100
   print('time step no: ',nn)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_1 & transform v_1 to a 3D array (file 1)
   u3d = np.load('datnpy/u3d_saved_'+str(nn)+'.npy')
   u3d_nfiles[:,:,:,n]= u3d
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_2 & transform v_2 to a 3D array (file 1)
   v3d = np.load('datnpy/v3d_saved_'+str(nn)+'.npy')
   v3d_nfiles[:,:,:,n]= v3d
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_3 & transform v_3 to a 3D array (file 1)
   w3d = np.load('datnpy/w3d_saved_'+str(nn)+'.npy')
   w3d_nfiles[:,:,:,n]= w3d

# merge nfiles. This means that new nk = nfiles*nk
u3d=u3d_nfiles[:,:,:,1]
v3d=v3d_nfiles[:,:,:,1]
w3d=w3d_nfiles[:,:,:,1]
for n in range(1,nfiles):
   u3d=np.concatenate((u3d, u3d_nfiles[:,:,:,n]), axis=2)
   v3d=np.concatenate((v3d, v3d_nfiles[:,:,:,n]), axis=2)
   w3d=np.concatenate((w3d, w3d_nfiles[:,:,:,n]), axis=2)





# x coordinate direction = index 0, first index
# y coordinate direction = index 1, second index
# z coordinate direction = index 2, third index



nk=len(u3d[0,0,:])
print('new nk including all files: ',nk)

cyclic_x=False
cyclic_z=True

fx,fy,areawx,areawy,areasx,areasy,vol= compute_geometry(x2d,y2d,xp2d,yp2d,nk,dz)

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

# compute 1st-order gradients
u_face_w,u_face_s,u_face_l=compute_face(u3d,'n','n','d','d','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
v_face_w,v_face_s,v_face_l=compute_face(v3d,'n','n','d','d','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
w_face_w,w_face_s,w_face_l=compute_face(w3d,'n','n','d','d','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)

dudx=dphidx(u_face_w,u_face_s,areawx,areawy,areasx,areasy,vol)
dudy=dphidy(u_face_w,u_face_s,areawx,areawy,areasx,areasy,vol)
dudz=dphidz(u_face_l,dz)

dvdx=dphidx(v_face_w,v_face_s,areawx,areawy,areasx,areasy,vol)
dvdy=dphidy(v_face_w,v_face_s,areawx,areawy,areasx,areasy,vol)
dvdz=dphidz(v_face_l,dz)

dwdx=dphidx(w_face_w,w_face_s,areawx,areawy,areasx,areasy,vol)
dwdy=dphidy(w_face_w,w_face_s,areawx,areawy,areasx,areasy,vol)
dwdz=dphidz(w_face_l,dz)

# compute 2nd-order gradients of U
u_face_w,u_face_s,u_face_l=compute_face(dudx,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2udx2=dphidx(u_face_w,u_face_s,areawx,areawy,areasx,areasy,vol)
d2udxy=dphidy(u_face_w,u_face_s,areawx,areawy,areasx,areasy,vol)
d2udxz=dphidz(u_face_l,dz)

u_face_w,u_face_s,u_face_l=compute_face(dudy,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2udy2=dphidy(u_face_w,u_face_s,areawx,areawy,areasx,areasy,vol)
d2udyz=dphidz(u_face_l,dz)

u_face_w,u_face_s,u_face_l=compute_face(dudz,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2udz2=dphidz(u_face_l,dz)

# compute 2nd-order gradients of V
v_face_w,v_face_s,v_face_l=compute_face(dvdx,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2vdx2=dphidx(v_face_w,v_face_s,areawx,areawy,areasx,areasy,vol)
d2vdxy=dphidy(v_face_w,v_face_s,areawx,areawy,areasx,areasy,vol)
d2vdxz=dphidz(v_face_l,dz)

v_face_w,v_face_s,v_face_l=compute_face(dvdy,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2vdy2=dphidy(v_face_w,v_face_s,areawx,areawy,areasx,areasy,vol)
d2vdyz=dphidz(v_face_l,dz)

v_face_w,v_face_s,v_face_l=compute_face(dvdz,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2vdz2=dphidz(v_face_l,dz)

# compute 2nd-order gradients of W
w_face_w,w_face_s,w_face_l=compute_face(dwdx,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2wdx2=dphidx(w_face_w,w_face_s,areawx,areawy,areasx,areasy,vol)
d2wdxy=dphidy(w_face_w,w_face_s,areawx,areawy,areasx,areasy,vol)
d2wdxz=dphidz(w_face_l,dz)

w_face_w,w_face_s,w_face_l=compute_face(dwdy,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2wdy2=dphidy(w_face_w,w_face_s,areawx,areawy,areasx,areasy,vol)
d2wdyz=dphidz(w_face_l,dz)

w_face_w,w_face_s,w_face_l=compute_face(dwdz,'n','n','n','n','n',x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk)
d2wdz2=dphidz(w_face_l,dz)

#User defined

#Plot instantaneous Von Karman length scale

Upp = np.sqrt(np.multiply(d2udx2 + d2udy2 + d2udz2,d2udx2 + d2udy2 + d2udz2) \
   + np.multiply(d2vdx2 + d2vdy2 + d2vdz2,d2vdx2 + d2vdy2 + d2vdz2) \
      + np.multiply(d2wdx2 + d2wdy2 + d2wdz2,d2wdx2 + d2wdy2 + d2wdz2))
kappa = 0.41
S11 = dudx
S12 = 1/2 * (dudy + dvdx)
S13 = 1/2 * (dudz + dwdx)
S22 = dvdy
S23 = 1/2 * (dvdz + dwdy)
S33 = dwdz

s = np.sqrt(S11**2 + 2 * S12**2 + 2 * S13**2 + S22**2 + 2 * S23**2 + S33**2)

L_vk_inst = kappa * np.divide(s,Upp)
#Spanwise average
L_vk_inst_av = 1/len(L_vk_inst[1,1,:]) * L_vk_inst.sum(axis=2)

C_des = 0.61
calcDelta = False
if calcDelta:
    Delta = np.zeros((ni,nj))
    for i in range(ni):
        for j in range(nj):
            sidesize = np.zeros(3) #I'm lazy, I just assume that cells are kind of isotropixc
            sidesize[0] = (x2d[i+1,j]-x2d[i,j])**2 + (y2d[i+1,j]-y2d[i,j])**2
            sidesize[1] = (x2d[i,j+1]-x2d[i,j])**2 + (y2d[i,j+1]-y2d[i,j])**2
            sidesize[2] = dz**2
            Delta[i,j] = np.sqrt(np.max(sidesize))
            np.disp("Cell " + str(i * nj + j) + " out of " + str(ni * nj - 1) + " done!")
    
    # Write Walldistance vector to file
    np.savetxt("datnpy/Delta.dat", Delta)
else:
    if os.path.isfile("datnpy/Delta.dat"):
        # Read Walldistance vector from file
        Delta = np.loadtxt("datnpy/Delta.dat")
    else:
        print("Error: Delta file not found.")

#Calculate time averaged quantities for steady von Karman
itstep,nk_temp,dz=np.load('datnpy/itstep-hump-IDDES.npy')
u2d=np.load('datnpy/u_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
v2d=np.load('datnpy/v_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity

cyclic_x=False
cyclic_z=True

fx2d,fy2d,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d= compute_geometry_2d(x2d,y2d,xp2d,yp2d)

dudxS= np.zeros((ni,nj))
dudyS= np.zeros((ni,nj))
dvdxS= np.zeros((ni,nj))
dvdyS= np.zeros((ni,nj))

u_face_w,u_face_s=compute_face_2d(u2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
v_face_w,v_face_s=compute_face_2d(v2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)

dudxS=dphidx_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dudyS=dphidy_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dvdxS=dphidx_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dvdyS=dphidy_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

# compute 2nd-order gradients of U
u_face_w,u_face_s=compute_face_2d(dudxS,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
d2udx2S=dphidx_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

u_face_w,u_face_s=compute_face_2d(dudyS,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
d2udy2S=dphidy_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

# compute 2nd-order gradients of V
v_face_w,v_face_s=compute_face_2d(dvdxS,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
d2vdx2S=dphidx_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

v_face_w,v_face_s=compute_face_2d(dvdyS,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
d2vdy2S=dphidy_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

UppS = np.sqrt(np.multiply(d2udx2S + d2udy2S,d2udx2S + d2udy2S) \
   + np.multiply(d2vdx2S + d2vdy2S,d2vdx2S + d2vdy2S))

S11 = dudxS
S12 = 1/2 * (dudyS + dvdxS)
S22 = dvdyS

sSteady = np.sqrt(2*(S11**2 + 2 * S12**2 + 2 * S22**2))

L_vk_Steady = kappa * np.divide(sSteady,UppS)

#Calculate RANS length scale
itstep,nk_temp,dz_temp=np.load('datnpy/itstep-hump-IDDES.npy')
k_model2d=np.load('datnpy/k_averaged-hump-IDDES.npy')/itstep
eps2d=np.load('datnpy/eps_averaged-hump-IDDES.npy')/itstep
L_RANS = np.divide(k_model2d**(3/2),eps2d)

stations = [0.66, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("Length scales")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-xp2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(L_vk_inst_av[iinner,:],yp2d[iinner,:],'b-',label="L_{vk,inst}")
    ax[i].plot(L_vk_Steady[iinner,:],yp2d[iinner,:],'r-',label="L_{vk,steady}")
    ax[i].plot(C_des * Delta[iinner,:],yp2d[iinner,:],'g-',label="C_{DES}\Delta")
    ax[i].plot(L_RANS[iinner,:],yp2d[iinner,:],'m-',label="L_{RANS}")

    ax[i].set_title("$x$ = " + str(stations[i]))
    ax[i].legend()
plt.savefig("img/LengthScales.eps")

#Plot source terms
zeta = 1.5
s_av = 1/len(s[1,1,:]) * s.sum(axis=2)
P_SAS_inst = zeta * kappa * np.multiply(s_av**2,np.divide(L_RANS,L_vk_inst_av))
P_SAS_steady = zeta * kappa * np.multiply(s_av**2,np.divide(L_RANS,L_vk_Steady))

stations = [0.66, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("Production terms P_\omega")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-xp2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(P_SAS_inst[iinner,:],yp2d[iinner,:],'b-',label="inst")
    ax[i].plot(P_SAS_steady[iinner,:],yp2d[iinner,:],'r-',label="steady")

    ax[i].set_title("$x$ = " + str(stations[i]))
    ax[i].legend()
plt.savefig("img/Productions.eps")

xstations = [0.65, 0.8, 1.1, 1.3]
jystations = [3, 10, 30, 70]
zmax, nk=np.loadtxt('datnpy/z_hump_IDDES.dat')
nk=int(nk)
two_corrz=np.zeros((nk,nfiles,len(xstations),len(jystations)))
for iouter in range(0,len(xstations)):
    i = (np.abs(xstations[iouter]-xp2d[:,1])).argmin()
    for jouter in range(0,len(jystations)):
      j = jystations[jouter]
      for n in range(0,nfiles):
         for k in range(0,nk):
            two_corrz[k,n,iouter,jouter]=two_corrz[k,n,iouter,jouter]+circcorrcoef(w3d[i,j,:],np.roll(w3d[i,j,:],k))/nk

corr = two_corrz[:,1,2,2] / max(two_corrz[:,1,2,2])
fig,ax = plt.subplots()
plt.plot(zp2d, corr)
plt.savefig("img/TwoPointCorrellation.eps")

plt.show(block=True)