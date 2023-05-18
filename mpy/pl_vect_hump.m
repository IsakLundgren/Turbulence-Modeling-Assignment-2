% Diffuser, lada
close all
clear all
format long

re =9.36e+5;
viscos =1/re;

load xy_hump.dat
x=xy_hump(:,1);
y=xy_hump(:,2);

ni=314;
nj=122;
nim1=ni-1;
njm1=nj-1;
load vectz_aiaa_paper.dat
tec=vectz_aiaa_paper;;
%load vectz_aiaa_journal.dat
%tec=vectz_aiaa_journal;;
%load vectz_zonal_pans.dat
%tec=vectz_zonal_pans;;
n=length(tec);
ntstep=tec(1);


%            write(48,*)uvec(i,j)
%            write(48,*)vvec(i,j)
%            write(48,*)dummy(i,j)
%            write(48,*)uvec2(i,j)
%            write(48,*)vvec2(i,j)
%            write(48,*)wvec2(i,j)
%            write(48,*)uvvec(i,j)
%            write(48,*)p2D(i,j)
%            write(48,*)rk2D(i,j)
%            write(48,*)vis2D(i,j)  
%            write(48,*)dissp2D(i,j)
%            write(48,*)uvturb(i,j)

nn=12;
ju=2:nn:n;
jv=3:nn:n;
juu=5:nn:n;
jvv=6:nn:n;
jww=7:nn:n;
juv=8:nn:n;
jp=9:nn:n;
jk=10:nn:n;
jvis=11:nn:n;
jdiss=12:nn:n;
juv_model=13:nn:n;


u=tec(ju)/ntstep;
v=tec(jv)/ntstep;
uu=tec(juu)/ntstep;
vv=tec(jvv)/ntstep;
ww=tec(jww)/ntstep;
uv=tec(juv)/ntstep;
p=tec(jp)/ntstep;
k_model=tec(jk)/ntstep;
vis=tec(jvis)/ntstep;
diss=tec(jdiss)/ntstep;
uv_model=tec(juv_model)/ntstep;

% uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u.^2;
vv=vv-v.^2;
uv=uv-u.*v;

p_2d=reshape(p,nj,ni)';
uu_2d=reshape(uu,nj,ni)';
vv_2d=reshape(vv,nj,ni)';
ww_2d=reshape(ww,nj,ni)';
uv_2d=reshape(uv,nj,ni)';
k_model_2d=reshape(k_model,nj,ni)';
uv_model_2d=reshape(uv_model,nj,ni)';
diss_2d=reshape(diss,nj,ni)';  
vis_2d=reshape(vis,nj,ni)';%this is to total viscosity, i.e. vis_tot=vis+vis_turb
u_2d=reshape(u,nj,ni)';
v_2d=reshape(v,nj,ni)';
x_2d=reshape(x,ni,nj);
y_2d=reshape(y,ni,nj);


%variables=x,y,u,v,uu,vv,uv
load x065_off.dat

[dudx_2d,dudy_2d] = dphidx_dy(x_2d,y_2d,u_2d,ni,nj);


%************
% profile of u
figure(1)
xx=0.65;
ii = find(x_2d(:,1) < xx);
i=ii(end);
plot(u_2d(i,:),y_2d(i,:),'linewidth',2)
hold
plot(x065_off(:,3),x065_off(:,2),'o','linewidth',2)

h=gca
xlabel('U','fontsize',[20])
ylabel('y/H','fontsize',[20])
h1=gca;
axis([0 1.3 0.1 0.3])
set(h1,'fontsize',[20])
title('x=0.65','fontsize',[20])
h=gca;
set(h,'fontsize',[20]) %20-pt text
print -depsc u_prof_aiaa_paper.ps


%************
% profile of uu
figure(2)
xx=0.65;
ii = find(x_2d(:,1) < xx);
i=ii(end);
plot(uu_2d(i,:),y_2d(i,:),'linewidth',2)
hold
plot(x065_off(:,5),x065_off(:,2),'o','linewidth',2)

h=gca
xlabel('vv','fontsize',[20])
ylabel('y/H','fontsize',[20])
title('x=0.65','fontsize',[20])
h1=gca;
axis([0 0.010 0.1 0.3])
set(h1,'fontsize',[20])
h=gca;
set(h,'fontsize',[20]) %20-pt text
print -depsc vv_prof_aiaa_paper.ps



%************
% profile of uu
figure(3)
xx=0.65;
ii = find(x_2d(:,1) < xx);
i=ii(end);
plot(uu_2d(i,:),y_2d(i,:),'linewidth',2)
hold

h=gca
xlabel('uu','fontsize',[20])
ylabel('y/H','fontsize',[20])
h1=gca;
axis([0 0.05 0 0.4])
set(h1,'fontsize',[20])
h=gca;
set(h,'fontsize',[20]) %20-pt text
print -depsc uu_prof_aiaa_paper.ps


%************
% contour of uu
figure(4)
hs=contour(x_2d,y_2d,uu_2d,20);
hold
j=1;
plot(x_2d(:,j),y_2d(:,j))
j=nj-1
plot(x_2d(:,j),y_2d(:,j))
i=1;
plot(x_2d(i,:),y_2d(i,:))

h=gca
xlabel('x/H','fontsize',[20])
ylabel('y/H','fontsize',[20])
h1=gca;
colorbar
axis([0.6 4 0 1])
axis equal
set(h1,'fontsize',[20])
h=gca;
set(h,'fontsize',[20]) %20-pt text
print -depsc uuiso_aiaa_paper.ps

%************
% velocity vector plot
figure(5)

ii=4;
i=1:ii:ni; % plot every fourth i-node
j=1:ii:nj; % plot every fourth j-node
ss=1.2; %vector length
ss1=0; % plot the arrows without the automatic scaling
quiver(x_2d(i,j),y_2d(i,j),ss*u_2d(i,j),ss*v_2d(i,j),ss);
hold
h=gca;
title('vect-xy')
xlabel('x')
ylabel('y')
axis([0.6 4 0 1])
axis equal
h=gca;
set(h,'fontsize',[20]) %20-pt text
print -deps vect_aiaa_paper.ps

%%%%%%%%%%%%%%%%%%%%%% grid
figure(6)
for i=1:ii:ni
   plot(x_2d(i,:),y_2d(i,:))
   if i== 1
      hold
   end
end
for j=1:ii:nj
   plot(x_2d(:,j),y_2d(:,j))
end
axis([0.6 4 0 1])
axis([0.6 2 0 1])
axis equal
title('grid')
xlabel('x')
ylabel('y')
h=gca;
%set(h,'xtickl',[0.6 1 2 3 4])
set(h,'xtick',[0.6 1 2 3 4])
set(h,'fontsize',[20])
%set(h,'ytickl',[])
print -deps grid_aiaa_paper.ps

