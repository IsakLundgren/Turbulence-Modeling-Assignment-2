% Diffuser, lada
close all
clear all
format long

re =9.36e+5;
viscos =1/re;

load xy_hump_fine.dat
x1=xy_hump_fine(:,1);
y1=xy_hump_fine(:,2);





nim1=x1(1);
njm1=y1(1);

ni=nim1+1;
nj=njm1+1;


x=x1(2:end);
y=y1(2:end);

x_2d=reshape(x,nim1,njm1);
y_2d=reshape(y,nim1,njm1);

% compute cell centers
xp2d=zeros(ni,nj);
yp2d=zeros(ni,nj);

for jj=1:nj
for ii=1:ni

    im1=max(ii-1,1);
    jm1=max(jj-1,1);

    i=min(ii,nim1);
    j=min(jj,njm1);


   xp2d(ii,jj)=0.25*(x_2d(i,j)+x_2d(im1,j)+x_2d(i,jm1)+x_2d(im1,jm1));
   yp2d(ii,jj)=0.25*(y_2d(i,j)+y_2d(im1,j)+y_2d(i,jm1)+y_2d(im1,jm1));
end
end


load vectz_fine.dat
vectz=vectz_fine;
ntstep=vectz(1);
ni=vectz(2);
nj=vectz(3);
nk=vectz(4);
n=length(vectz);

%            write(48,*)uvec(i,j)
%            write(48,*)vvec(i,j)
%            write(48,*)fk2d(i,j)
%            write(48,*)uvec2(i,j)
%            write(48,*)vvec2(i,j)
%            write(48,*)wvec2(i,j)
%            write(48,*)uvvec(i,j)
%            write(48,*)p2d(i,j)
%            write(48,*)rk2d(i,j)
%            write(48,*)vis2d(i,j)
%            write(48,*)dissp2d(i,j)
%            write(48,*)wvec(i,j)
%            write(48,*)vtvec(i,j)
%            write(48,*)tvec(i,j)


nn=14;
nst=3;
iu=nst+2:nn:n;
iv=nst+3:nn:n;
ifk=nst+4:nn:n;
iuu=nst+5:nn:n;
ivv=nst+6:nn:n;
iww=nst+7:nn:n;
iuv=nst+8:nn:n;
ip=nst+9:nn:n;
ik=nst+10:nn:n;
ivis=nst+11:nn:n;
idiss=nst+12:nn:n;



u=vectz(iu)/ntstep;
v=vectz(iv)/ntstep;
fk=vectz(ifk)/ntstep;
uu=vectz(iuu)/ntstep;
vv=vectz(ivv)/ntstep;
ww=vectz(iww)/ntstep;
uv=vectz(iuv)/ntstep;
p=vectz(ip)/ntstep;
k_model=vectz(ik)/ntstep;
vis=vectz(ivis)/ntstep;
diss=vectz(idiss)/ntstep;

% uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u.^2;
vv=vv-v.^2;
uv=uv-u.*v;

p_2d=reshape(p,nj,ni)';
uu_2d=reshape(uu,nj,ni)';
vv_2d=reshape(vv,nj,ni)';
fk_2d=reshape(fk,nj,ni)';
ww_2d=reshape(ww,nj,ni)';
uv_2d=reshape(uv,nj,ni)';
k_model_2d=reshape(k_model,nj,ni)';
diss_2d=reshape(diss,nj,ni)';  %this is to total viscosity, i.e. vis_tot=vis+vis_turb
vis_2d=reshape(vis,nj,ni)';
u_2d=reshape(u,nj,ni)';
v_2d=reshape(v,nj,ni)';

% set fk_2d=1 at upper boundary
fk_2d(:,nj)=fk_2d(:,nj-1);

load x065_off.dat

[dudx_2d,dudy_2d] = dphidx_dy(xp2d,yp2d,u_2d,ni,nj);

%*************************
% plot u
figure(2)
xx=0.65;
ii = find(xp2d(:,1) < xx);
i1=ii(end);
plot(u_2d(i1,:),yp2d(i1,:)-yp2d(i1,1),'b-','linewidth',2)
hold
h1=plot(x065_off(:,3),x065_off(:,2)-yp2d(i1,1),'bo','linewidth',2)
set(h1,'MarkerFaceColor','white')

xlabel('x')
ylabel('y')
title('065')
axis([0 1.3 0 0.2])
h=gca;
set(h,'fontsize',[20])
% zoom. 'Position',[left bottom width height])
% create smaller axes in top right, and plot on it
%axes('Position',[.1 .4 0.4 0.4])
axes('OuterPosition',[.17 .46 0.55 0.45])
plot(u_2d(i1,:),yp2d(i1,:)-yp2d(i1,1),'b-','linewidth',2)
hold
h1=plot(x065_off(:,3),x065_off(:,2)-yp2d(i1,1),'bo','linewidth',2)
set(h1,'MarkerFaceColor','white')

axis([0 1.2 0. 0.004])

print u065_hump_from_vectz.ps  -depsc

