import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})

re =9.36e+5
viscos =1/re

xy_hump_fine = np.loadtxt("xy_hump.dat")
x=xy_hump_fine[:,0]
y=xy_hump_fine[:,1]

ni=314
nj=122

nim1=ni-1
njm1=nj-1

# read data file
vectz=np.genfromtxt("vectz_aiaa_journal.dat",comments="%")
ntstep=vectz[0]
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)dummy(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2D(i,j)
#            write(48,*)rk2D(i,j)
#            write(48,*)vis2D(i,j)  
#            write(48,*)dissp2D(i,j)
#            write(48,*)uvturb(i,j)



nn=12
nst=0
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
iuv_model=range(nst+12,n,nn)

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
uv_model=vectz[iuv_model]/ntstep

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
x_2d=np.transpose(np.reshape(x,(nj,ni)))
y_2d=np.transpose(np.reshape(y,(nj,ni)))

# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

x065_off=np.genfromtxt("x065_off.dat",comments="%")

# the funtion dphidx_dy wants x and y arrays to be one cell smaller than u2d. Hence I take away the last row and column below
x_2d_new=np.delete(x_2d,-1,0)
x_2d_new=np.delete(x_2d_new,-1,1)
y_2d_new=np.delete(y_2d,-1,0)
y_2d_new=np.delete(y_2d_new,-1,1)
# compute the gradient
dudx,dudy=dphidx_dy(x_2d_new,y_2d_new,u_2d)


#*************************
# plot u
fig1 = plt.figure("Figure 1")
plt.clf() #clear the figure
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u_2d[i1,:],y_2d[i1,:]-y_2d[i1,0],'b-')
plt.plot(x065_off[:,2],x065_off[:,1]-y_2d[i1,0],'bo')
plt.xlabel("$U$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 1.3,0,0.2])
plt.savefig('u065_hump_python.eps')

#*************************
# plot vv
fig2 = plt.figure("Figure 2")
plt.clf() #clear the figure
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(vv_2d[i1,:],y_2d[i1,:]-y_2d[i1,0],'b-')
plt.plot(x065_off[:,5],x065_off[:,1]-y_2d[i1,0],'bo')
plt.xlabel("$\overline{v'v'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.01,0,0.2])
plt.savefig('vv065_hump_python.eps')

#*************************
# plot uu
fig3 = plt.figure("Figure 3")
plt.clf() #clear the figure
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(vv_2d[i1,:],y_2d[i1,:]-y_2d[i1,0],'b-')
plt.plot(x065_off[:,5],x065_off[:,1]-y_2d[i1,0],'bo')
plt.xlabel("$\overline{u'u'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.05,0,0.2])
plt.savefig('uu065_hump_python.eps')

################################ contoue plot
fig4 = plt.figure("Figure 4")
plt.clf() #clear the figure
plt.contourf(x_2d,y_2d,uu_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.clim(0,0.05)
plt.axis([0.6,1.5,0,1])
plt.title("contour $\overline{u'u'}$")
plt.savefig('piso_python.eps')

################################ vector plot
fig5 = plt.figure("Figure 5")
plt.clf() #clear the figure
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(x_2d[::k,::k],y_2d[::k,::k],u_2d[::k,::k],v_2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis([0.6,1.5,0,1])
plt.title("vector plot")
plt.savefig('vect_python.eps')


