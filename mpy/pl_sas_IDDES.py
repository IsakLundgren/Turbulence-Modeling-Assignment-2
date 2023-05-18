import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
from grad_xyz import dphidx_2d,dphidy_2d,compute_face_2d,compute_geometry_2d, \
                     dphidx,dphidy,dphidz,compute_face,compute_geometry
plt.rcParams.update({'font.size': 22})
plt.interactive(True)
re =9.36e+5
viscos =1/re

datax= np.loadtxt('x2d_hump_IDDES.dat')
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt('y2d_hump_IDDES.dat')
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
zmax, nk=np.loadtxt('z_hump_IDDES.dat')
nk=int(nk)
dz=zmax/nk

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
   u3d = np.load('u3d_saved_'+str(nn)+'.npy')
   u3d_nfiles[:,:,:,n]= u3d

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

