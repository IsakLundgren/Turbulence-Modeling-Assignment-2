import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dphidx_dy import dphidx_dy
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

re =9.36e+5
viscos =1/re

xy_hump_fine = np.loadtxt("xy_hump_fine.dat")
x1=xy_hump_fine[:,0]
y1=xy_hump_fine[:,1]


nim1=int(x1[0])
njm1=int(y1[0])

ni=nim1+1
nj=njm1+1


x=x1[1:]
y=y1[1:]

x_2d=np.reshape(x,(njm1,nim1))
y_2d=np.reshape(y,(njm1,nim1))

x_2d=np.transpose(x_2d)
y_2d=np.transpose(y_2d)

# compute cell centers
xp2d= np.zeros((ni,nj))
yp2d= np.zeros((ni,nj))

for jj in range (0,nj):
   for ii in range (0,ni):

      im1=max(ii-1,0)
      jm1=max(jj-1,0)

      i=min(ii,nim1-1)
      j=min(jj,njm1-1)


      xp2d[ii,jj]=0.25*(x_2d[i,j]+x_2d[im1,j]+x_2d[i,jm1]+x_2d[im1,jm1])
      yp2d[ii,jj]=0.25*(y_2d[i,j]+y_2d[im1,j]+y_2d[i,jm1]+y_2d[im1,jm1])

# read data file
vectz = np.loadtxt("vectz_fine.dat")
ntstep=vectz[0]
ni=int(vectz[1])
nj=int(vectz[2])
nk=int(vectz[3])
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)fk2d(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2d(i,j)
#            write(48,*)rk2d(i,j)
#            write(48,*)vis2d(i,j)
#            write(48,*)dissp2d(i,j)
#            write(48,*)wvec(i,j)
#            write(48,*)vtvec(i,j)
#            write(48,*)tvec(i,j)


nn=14
nst=3
iu=range(nst+1,n,nn)
iv=range(nst+2,n,nn)
ifk=range(nst+3,n,nn)
iuu=range(nst+4,n,nn)
ivv=range(nst+5,n,nn)
iww=range(nst+6,n,nn)
iuv=range(nst+7,n,nn)
ip=range(nst+8,n,nn)
ik=range(nst+9,n,nn)
ivis=range(nst+10,n,nn)
idiss=range(nst+11,n,nn)

u=vectz[iu]/ntstep
v=vectz[iv]/ntstep
fk=vectz[ifk]/ntstep
uu=vectz[iuu]/ntstep
vv=vectz[ivv]/ntstep
ww=vectz[iww]/ntstep
uv=vectz[iuv]/ntstep
p=vectz[ip]/ntstep
k_model=vectz[ik]/ntstep
vis=vectz[ivis]/ntstep
diss=vectz[idiss]/ntstep

# uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u**2
vv=vv-v**2
uv=uv-u*v

p_2d=np.reshape(p,(ni,nj))
u_2d=np.reshape(u,(ni,nj))
v_2d=np.reshape(v,(ni,nj))
fk_2d=np.reshape(fk,(ni,nj))
uu_2d=np.reshape(uu,(ni,nj))
uv_2d=np.reshape(uv,(ni,nj))
vv_2d=np.reshape(vv,(ni,nj))
ww_2d=np.reshape(ww,(ni,nj))
k_model_2d=np.reshape(k_model,(ni,nj))
vis_2d=np.reshape(vis,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb
diss_2d=np.reshape(diss,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb

# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

dz=0.2/nk

x065_off=np.genfromtxt("x065_off.dat", dtype=None,comments="%")

# compute the gradient
dudx,dudy=dphidx_dy(x_2d,y_2d,u_2d)


#*************************
# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u_2d[i1,:],yp2d[i1,:],'b-')
plt.plot(x065_off[:,2],x065_off[:,1],'bo')
plt.xlabel("$U$")
plt.ylabel("$y$")
plt.title("$x=0.65$")
plt.axis([0, 1.3,0.115,0.2])

# Create inset of width 30% and height 40% of the parent axes' bounding box
# at the lower left corner (loc=3)
# upper left corner (loc=2)
# use borderpad=1, i.e.
# 22 points padding (as 22pt is the default fontsize) to the parent axes
axins1 = inset_axes(ax1, width="40%", height="30%", loc=2, borderpad=1)
plt.plot(u_2d[i1,:],yp2d[i1,:],'b-')
plt.axis([0, 1.3,0.115,0.13])
# reduce fotnsize 
axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)

# Turn ticklabels of insets off
axins1.tick_params(labelleft=False, labelbottom=False)

plt.plot(x065_off[:,2],x065_off[:,1],'bo')

plt.savefig('u065_hump_python.eps',bbox_inches='tight')

