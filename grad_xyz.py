def dphidx_2d(phi_face_w,phi_face_s,areawx,areawy,areasx,areasy,vol):

   phi_w=phi_face_w[0:-1,:]*areawx[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawx[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasx[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasx[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidy_2d(phi_face_w,phi_face_s,areawx,areawy,areasx,areasy,vol):

   phi_w=phi_face_w[0:-1,:]*areawy[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawy[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasy[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasy[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidx(phi_face_w,phi_face_s,areawx,areawy,areasx,areasy,vol):

   phi_w=phi_face_w[0:-1,:,:]*areawx[0:-1,:,:]
   phi_e=-phi_face_w[1:,:,:]*areawx[1:,:,:]
   phi_s=phi_face_s[:,0:-1,:]*areasx[:,0:-1,:]
   phi_n=-phi_face_s[:,1:,:]*areasx[:,1:,:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidy(phi_face_w,phi_face_s,areawx,areawy,areasx,areasy,vol):

   phi_w=phi_face_w[0:-1,:,:]*areawy[0:-1,:,:]
   phi_e=-phi_face_w[1:,:,:]*areawy[1:,:,:]
   phi_s=phi_face_s[:,0:-1,:]*areasy[:,0:-1,:]
   phi_n=-phi_face_s[:,1:,:]*areasy[:,1:,:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidz(phi_face_l,dz):

   phi_l=phi_face_l[:,:,0:-1]
   phi_h=phi_face_l[:,:,1:]
   return (phi_h-phi_l)/dz

def compute_face(phi3d,phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type,phi_bc_z_type,\
    x2d,y2d,fx,fy,cyclic_x,cyclic_z,ni,nj,nk):
   import numpy as np

   phi3d_face_w=np.empty((ni+1,nj,nk))
   phi3d_face_s=np.empty((ni,nj+1,nk))
   phi3d_face_l=np.empty((ni,nj,nk+1))
   phi3d_face_w[1:,:,:]=fx*np.roll(phi3d,-1,axis=0)+(1-fx)*phi3d
   phi3d_face_s[:,1:,:]=fy*np.roll(phi3d,-1,axis=1)+(1-fy)*phi3d
   phi3d_face_l[:,:,1:]=0.5*np.roll(phi3d,-1,axis=2)+0.5*phi3d

# west boundary 
   phi3d_face_w[0,:,:]=0
   if phi_bc_west_type == 'n': 
# neumann
      phi3d_face_w[0,:,:]=phi3d[0,:,:]
   if cyclic_x:
      phi3d_face_w[0,:,:]=0.5*(phi3d[0,:,:]+phi3d[-1,:,:])


# east boundary 
   phi3d_face_w[-1,:,:]=0
   if phi_bc_east_type == 'n': 
# neumann
      phi3d_face_w[-1,:,:]=phi3d[-1,:,:]
   if cyclic_x:
      phi3d_face_w[-1,:,:]=0.5*(phi3d[0,:,:]+phi3d[-1,:,:])

# south boundary 
   phi3d_face_s[:,0,:]=0
   if phi_bc_south_type == 'n': 
# neumann
      phi3d_face_s[:,0,:]=phi3d[:,0,:]

# north boundary 
   phi3d_face_s[:,-1,:]=0
   if phi_bc_north_type == 'n': 
# neumann
      phi3d_face_s[:,-1,:]=phi3d[:,-1,:]

# low boundary 
   phi3d_face_l[:,:,0]=0
# high boundary 
   phi3d_face_l[:,:,-1]=0
   if phi_bc_z_type == 'n': 
# neumann
# low boundary 
      phi3d_face_l[:,:,0]= phi3d[:,:,0]
# high boundary 
      phi3d_face_l[:,:,-1]= phi3d[:,:,-1]
   if cyclic_z:
# low boundary 
      phi3d_face_l[:,:,0]= 0.5*(phi3d[:,:,-1]+phi3d[:,:,0])
# high boundary 
      phi3d_face_l[:,:,-1]= 0.5*(phi3d[:,:,-1]+phi3d[:,:,0])
   
   return phi3d_face_w,phi3d_face_s,phi3d_face_l

def compute_face_2d(phi2d,phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type,\
    x2d,y2d,fx,fy,cyclic_x,ni,nj):
   import numpy as np

   phi2d_face_w=np.empty((ni+1,nj))
   phi2d_face_s=np.empty((ni,nj+1))
   phi2d_face_w[1:,:]=fx*np.roll(phi2d,-1,axis=0)+(1-fx)*phi2d
   phi2d_face_s[:,1:]=fy*np.roll(phi2d,-1,axis=1)+(1-fy)*phi2d

# west boundary 
   phi2d_face_w[0,:]=0
   if phi_bc_west_type == 'n': 
# neumann
      phi2d_face_w[0,:]=phi2d[0,:]
   if cyclic_x:
      phi2d_face_w[0,:]=0.5*(phi2d[0,:]+phi2d[-1,:])


# east boundary 
   phi2d_face_w[-1,:]=0
   if phi_bc_east_type == 'n': 
# neumann
      phi2d_face_w[-1,:]=phi2d[-1,:]
   if cyclic_x:
      phi2d_face_w[-1,:]=0.5*(phi2d[0,:]+phi2d[-1,:])

# south boundary 
   phi2d_face_s[:,0]=0
   if phi_bc_south_type == 'n': 
# neumann
      phi2d_face_s[:,0]=phi2d[:,0]

# north boundary 
   phi2d_face_s[:,-1]=0
   if phi_bc_north_type == 'n': 
# neumann
      phi2d_face_s[:,-1]=phi2d[:,-1]

   return phi2d_face_w,phi2d_face_s

def compute_geometry_2d(x2d,y2d,xp2d,yp2d):
  import numpy as np

#  west face coordinate
  xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
  yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])
  
  del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
  del2x=((xw-np.roll(xp2d,1,axis=0))**2+(yw-np.roll(yp2d,1,axis=0))**2)**0.5
  fx=del2x/(del1x+del2x)
  fx2d = fx

#  south face coordinate
  xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
  ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

  del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
  del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
  fy=del2y/(del1y+del2y)
  fy2d=fy

  areawy=np.diff(x2d,axis=1)
  areawx=-np.diff(y2d,axis=1)
  areawx_2d= areawx
  areawy_2d= areawy

  areasy=-np.diff(x2d,axis=0)
  areasx=np.diff(y2d,axis=0)
  areasx_2d= areasx
  areasy_2d= areasy

  areaw=(areawx**2+areawy**2)**0.5
  areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
  ax=np.diff(x2d,axis=1)
  ay=np.diff(y2d,axis=1)
  bx=np.diff(x2d,axis=0)
  by=np.diff(y2d,axis=0)

  areaz_1=0.5*np.absolute(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

  ax=np.diff(x2d,axis=1)
  ay=np.diff(y2d,axis=1)
  areaz_2=0.5*np.absolute(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])

  areaz=areaz_1+areaz_2
  vol=areaz
  vol_2d=vol

  return fx2d,fy2d,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d

def compute_geometry(x2d,y2d,xp2d,yp2d,nk,dz):
  import numpy as np
#  west face coordinate
  xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
  yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])
  
  del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
  del2x=((xw-np.roll(xp2d,1,axis=0))**2+(yw-np.roll(yp2d,1,axis=0))**2)**0.5
  fx=del2x/(del1x+del2x)
  fx = np.dstack([fx]*nk)

#  south face coordinate
  xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
  ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

  del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
  del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
  fy=del2y/(del1y+del2y)
  fy = np.dstack([fy]*nk)

  areawy=np.diff(x2d,axis=1)*dz
  areawx=-np.diff(y2d,axis=1)*dz
  areawx_2d= areawx
  areawy_2d= areawy

# make them 3d
  areawx= np.dstack([areawx]*nk)
  areawy= np.dstack([areawy]*nk)

  areasy=-np.diff(x2d,axis=0)*dz
  areasx=np.diff(y2d,axis=0)*dz
  areasx_2d= areasx
  areasy_2d= areasy

# make them 3d
  areasx= np.dstack([areasx]*nk)
  areasy= np.dstack([areasy]*nk)

#  areaz=np.zeros((ni,nj,nk+1))

  areaw=(areawx**2+areawy**2)**0.5
  areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
  ax=np.diff(x2d,axis=1)
  ay=np.diff(y2d,axis=1)
  bx=np.diff(x2d,axis=0)
  by=np.diff(y2d,axis=0)

  areaz_1=0.5*np.absolute(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

  ax=np.diff(x2d,axis=1)
  ay=np.diff(y2d,axis=1)
  areaz_2=0.5*np.absolute(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])

  areaz=areaz_1+areaz_2
  vol=areaz*dz
  vol_2d=vol

# make it 3d
  vol= np.dstack([vol]*nk)

# make it 3d
  areaz= np.dstack([areaz]*(nk+1))


  return fx,fy,areawx,areawy,areasx,areasy,vol

