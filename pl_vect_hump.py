import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

re =9.36e+5
viscos =1/re

xy_hump_fine = np.loadtxt("datnpy/xy_hump.dat")
x=xy_hump_fine[:,0]
y=xy_hump_fine[:,1]

ni=314
nj=122

nim1=ni-1
njm1=nj-1

# read data file, load either paper or journal
vectz=np.genfromtxt("datnpy/vectz_aiaa_paper.dat",comments="%")
# vectz=np.genfromtxt("datnpy/vectz_aiaa_journal.dat",comments="%")
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
diss_2d=np.reshape(diss,(ni,nj)) 
uv_model_2d=np.reshape(uv_model,(ni,nj)) #User added
x_2d=np.transpose(np.reshape(x,(nj,ni)))
y_2d=np.transpose(np.reshape(y,(nj,ni)))

# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

x065_off=np.genfromtxt("datnpy/x065_off.dat",comments="%")

# the funtion dphidx_dy wants x and y arrays to be one cell smaller than u2d. Hence I take away the last row and column below
x_2d_new=np.delete(x_2d,-1,0)
x_2d_new=np.delete(x_2d_new,-1,1)
y_2d_new=np.delete(y_2d,-1,0)
y_2d_new=np.delete(y_2d_new,-1,1)
# compute the gradient
dudx,dudy=dphidx_dy(x_2d_new,y_2d_new,u_2d)
dvdx,dvdy=dphidx_dy(x_2d_new,y_2d_new,v_2d)


#*************************
# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,2],x065_off[:,1],'bo')
plt.xlabel("$U$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 1.3,0,0.3])
plt.savefig('img/u065_hump_python.eps')

#*************************
# plot vv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,5],x065_off[:,1],'bo')
plt.xlabel("$\overline{v'v'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.01,0,0.3])
plt.savefig('img/vv065_hump_python.eps')

#*************************
# plot uu
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(uu_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,4],x065_off[:,1],'bo')
plt.xlabel("$\overline{u'u'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.05,0,0.3])
plt.savefig('img/uu065_hump_python.eps')

################################ contour plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x_2d,y_2d,uu_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.clim(0,0.05)
plt.axis([0.6,1.5,0,1])
plt.title("contour $\overline{u'u'}$")
plt.savefig('img/piso_python.eps')

################################ vector plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(x_2d[::k,::k],y_2d[::k,::k],u_2d[::k,::k],v_2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis([0.6,1.5,0,1])
plt.title("vector plot")
plt.savefig('img/vect_python.eps')

##################################User defined code

#Plot vv for different x stations
path = "datnpy/"
stations = [0.66, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
filename = ["x066_off.dat", "x080_off.dat", "x090_off.dat", "x100_off.dat", "x110_off.dat", "x120_off.dat", "x130_off.dat"]

fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\overline{v'v'}$ for different $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(vv_2d[iinner,:],y_2d[iinner,:],'b-')

    x_off = np.genfromtxt(path + filename[i],comments="%")
    ax[i].plot(x_off[:,5],x_off[:,1],'bo')
    ax[i].set_title("$x$ = " + str(stations[i]))
fig.savefig('img/vvall_hump.eps')

#Plot uv and uv_model for 2 stations
stations = [0.65, 1]
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\overline{u'v'}$ for 2 $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    iturb = np.where((vis_2d[iinner,:]-viscos)/viscos >= 1) #Find indicies where the flow is turbulent MIGHT NOT BE NECESSARY
    ax[i].plot(uv_2d[iinner,:],y_2d[iinner,:],'b-',label="Resolved")
    ax[i].plot(uv_model_2d[iinner,:],y_2d[iinner,:],'r-',label="Modeled")
    ax[i].set_title("$x$ = " + str(stations[i]))
    ax[i].legend()
fig.savefig('img/uvandmodel.eps')

#Plot viscocity ratio
vis_rat_2d = (vis_2d-viscos)/viscos
stations = [0.65, 1]
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\\nu _t/\\nu$ for 2 $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(vis_rat_2d[iinner,:],y_2d[iinner,:],'b-')
    ax[i].set_title("$x$ = " + str(stations[i]))
fig.savefig('img/viscousratio.eps')

#Plot modeled stress gradients to resolved ones (ONLY IN X-Direction) 
# TODO take a look here again, resolved looks very small in comparison

# compute the model gradient
dudxnutdx,dudxnutdy=dphidx_dy(x_2d_new,y_2d_new,np.multiply(dudx,vis_2d-viscos))
dudynutdx,dudynutdy=dphidx_dy(x_2d_new,y_2d_new,np.multiply(dudy,vis_2d-viscos))

modelstressgrad = dudxnutdx + dudynutdy

#compute the resolved gradient
duudx, duudy = dphidx_dy(x_2d_new,y_2d_new,uu_2d)
duvdx, duvdy = dphidx_dy(x_2d_new,y_2d_new,uv_2d)

resolvedstressgrad = - duudx - duvdy

stations = [0.65, 1]
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("Modeled and resolved stress gradients for 2 $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(modelstressgrad[iinner,:],y_2d[iinner,:],'b-',label="Resolved")
    ax[i].plot(resolvedstressgrad[iinner,:],y_2d[iinner,:],'r-',label="Modeled")
    ax[i].set_title("$x$ = " + str(stations[i]))
    ax[i].legend()
fig.savefig('img/stressgradientcomparison.eps')

#Evaluate resolution
res1 = (vis_2d-viscos)/viscos

stations = [0.65, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\langle \\nu_t \\rangle / \\nu$ for different $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(res1[iinner,:],y_2d[iinner,:],'b-')
fig.savefig('img/res1.eps')

modeledshearstress = -np.multiply(dudy + dvdx,vis_2d-viscos)
res2 = np.abs(np.divide(modeledshearstress,modeledshearstress + uv_2d))
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("|$\langle \\tau_{12}\\rangle / (\langle \\tau_{12}\\rangle + \langle \\bar{v}_1'\\bar{v}_2'\\rangle)|$ for different $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(res2[iinner,:],y_2d[iinner,:],'b-')
fig.savefig('img/res2.eps')

k_res_2d = 1/2 * (uu_2d + vv_2d + ww_2d)
res3 = np.divide(k_model_2d, (k_model_2d + k_res_2d))
fig,ax = plt.subplots(1,len(stations), sharey=True)
fig.suptitle("$\langle k_{model}\\rangle / (\langle k_{model}\\rangle + k_{res})$ for different $x$ stations")
fig.supylabel("$y$")
for i in range(len(stations)):
    iinner = (np.abs(stations[i]-x_2d[:,1])).argmin()  # find index which closest fits xx
    ax[i].plot(res3[iinner,:],y_2d[iinner,:],'b-')
fig.savefig('img/res3.eps')

blt = np.zeros(ni)
dx = np.zeros(ni)
for i in range(ni-1):
    dx[i] = abs(x_2d[i,0] - x_2d[i-1,0])
    for j in range(nj-1):
        if res1[i,j] < 1:
            blt[i] = x_2d[i,j]
            break

dx[0] = dx[1]        

zmax, nk=np.loadtxt('datnpy/z_hump_IDDES.dat')
nk=int(nk)
dz=zmax/nk

res6 = np.zeros((ni,2))
res6[:,0] = np.divide(blt,dx)
res6[:,1] = np.divide(blt,dz)

fig,ax = plt.subplots(2,1, sharex=True)
fig.suptitle("Boundary layer thickness ratio")
fig.supxlabel("x")
ax[0].plot(x_2d[:,0],res6[:,0])
ax[1].plot(x_2d[:,1],res6[:,1])
ax[0].set_ylabel("$\delta/\Delta x$")
ax[1].set_ylabel("$\delta/\Delta z$")

plt.show(block=True)

