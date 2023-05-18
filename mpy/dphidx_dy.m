function [dphidx_p,dphidy_p] = dphidx_dy(x2d,y2d,phi_2d,ni,nj)



for i=2:ni-1
for j=2:nj-1

%%%%%%%%%%%% east face
   sx=x2d(i,j)-x2d(i,j-1);
   sy=y2d(i,j)-y2d(i,j-1);
% normalize
   d=sqrt(sx^2+sy^2);
   sx=sx/d;
   sy=sy/d;
   nx_e=sy;
   ny_e=-sx;
   d_e=d;
   phi_e=0.5*(phi_2d(i,j)+phi_2d(i,j-1));



%%%%%%%%%%%% west face
   sx=x2d(i-1,j)-x2d(i-1,j-1);
   sy=y2d(i-1,j)-y2d(i-1,j-1);
% normalize
   d=sqrt(sx^2+sy^2);
   sx=sx/d;
   sy=sy/d;
   nx_w=-sy;
   ny_w=sx;
   d_w=d;
   phi_w=0.5*(phi_2d(i-1,j)+phi_2d(i-1,j-1));

%%%%%%%%%%%% north face
   sx=x2d(i,j)-x2d(i-1,j);
   sy=y2d(i,j)-y2d(i-1,j);
% normalize
   d=sqrt(sx^2+sy^2);
   sx=sx/d;
   sy=sy/d;
   nx_n=-sy;
   ny_n=sx;
   d_n=d;
   phi_n=0.5*(phi_2d(i-1,j)+phi_2d(i,j));

%%%%%%%%%%%% south face
   sx=x2d(i,j-1)-x2d(i-1,j-1);
   sy=y2d(i,j-1)-y2d(i-1,j-1);
% normalize
   d=sqrt(sx^2+sy^2);
   sx=sx/d;
   sy=sy/d;
   nx_s=sy;
   ny_s=-sx;
   d_s=d;
   phi_s=0.5*(phi_2d(i-1,j-1)+phi_2d(i,j-1));

% area approaximated as the vector product of two triangles
   ax=x2d(i,j)-x2d(i,j-1);
   ay=y2d(i,j)-y2d(i,j-1);
   bx=x2d(i,j)-x2d(i-1,j);
   by=y2d(i,j)-y2d(i-1,j);
   area_p1=0.5*abs(ax*by-ay*bx);

   ax=x2d(i-1,j)-x2d(i-1,j-1);
   ay=y2d(i-1,j)-y2d(i-1,j-1);
   bx=x2d(i,j-1)-x2d(i-1,j-1);
   by=y2d(i,j-1)-y2d(i-1,j-1);
   area_p2=0.5*abs(ax*by-ay*bx);

   area_p=area_p1+area_p2;

% compute the gradient dudx, dudy at point P
   dphidx_p(i,j)=(phi_e*nx_e*d_e+phi_n*nx_n*d_n+phi_w*nx_w*d_w+phi_s*nx_s*d_s)/area_p;
   dphidy_p(i,j)=(phi_e*ny_e*d_e+phi_n*ny_n*d_n+phi_w*ny_w*d_w+phi_s*ny_s*d_s)/area_p;

end
end


% fix 2nd derivative = 0, i.e. 1st derivative constant
%dudy_1= dudy_2, (u_2-u_1)/0.5dy = (u_3-u_2)/dy =>2*(u_2-u_1) = (u_3-u_2) =>
% (3*u_2 - u_3)/2 = u_1
%
% on an non-equidistant grid we get
%
dphidx_p(:,1)=0;  % fix size on dphidx_p & dphidy_p
dphidx_p(:,nj)=0;
dphidx_p(1,:)=0;
dphidx_p(ni,:)=0;

dphidy_p(:,1)=0;
dphidy_p(:,nj)=0;
dphidy_p(1,:)=0;
dphidy_p(ni,:)=0;
dy1=y2d(:,2)-y2d(:,1);
dy2=y2d(:,3)-y2d(:,2);
dphidx_p(:,1)= dphidx_p(:,2) +(dphidx_p(:,2)-dphidx_p(:,3)).*dy1./dy2;
dphidy_p(:,1)= dphidy_p(:,2) +(dphidy_p(:,2)-dphidy_p(:,3)).*dy1./dy2;

dy1=y2d(:,nj)-y2d(:,nj-1);
dy2=y2d(:,nj-1)-y2d(:,nj-2);
dphidx_p(:,nj)= dphidx_p(:,nj-1) +(dphidx_p(:,nj-1)-dphidx_p(:,nj-2)).*dy1./dy2;
dphidy_p(:,nj)= dphidy_p(:,nj-1) +(dphidy_p(:,nj-1)-dphidy_p(:,nj-2)).*dy1./dy2;

% set cyclic in x direction
dphidx_p(1,:)= dphidx_p(ni-1,:);
dphidy_p(1,:)= dphidy_p(ni-1,:);
dphidx_p(ni,:)= dphidx_p(2,:);
dphidy_p(ni,:)= dphidy_p(2,:);

