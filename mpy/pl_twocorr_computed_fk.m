close all
clear all
format long

% x=0.65

load w_time_z65.dat

w_time=w_time_z65;
ntot=length(w_time);
nt=w_time(end);  %Number of time steps
nj=5; % j=10,30,50,70,90
nk=16; % k=1-16


i=81:81:ntot;

% remove the timestep number
w_time(i)=[];

w_y_z_t=reshape(w_time,nk,nj,nt);
%w_y_z_t_65= permute(w_y_z_t,[3 2 1]);
w_y_z_t_65= w_y_z_t;

% x=0.8

load w_time_z80.dat

w_time=w_time_z80;

i=81:81:ntot;

% remove the timestep number
w_time(i)=[];

w_y_z_t=reshape(w_time,nk,nj,nt);
%w_y_z_t_80= permute(w_y_z_t,[3 2 1]);
w_y_z_t_80= w_y_z_t;

% x=1.10

load w_time_z110.dat

w_time=w_time_z110;
ntot=length(w_time);

i=81:81:ntot;

% remove the timestep number
w_time(i)=[];

ntot=length(w_time);

w_y_z_t=reshape(w_time,nk,nj,nt);
%w_y_z_t_110= permute(w_y_z_t,[3 2 1]);
w_y_z_t_110= w_y_z_t;

% x=1.30

load w_time_z130.dat

w_time=w_time_z130;
ntot=length(w_time);

i=81:81:ntot;

% remove the timestep number
w_time(i)=[];

ntot=length(w_time);

w_y_z_t=reshape(w_time,nk,nj,nt);
%w_y_z_t_130= permute(w_y_z_t,[3 2 1]);
w_y_z_t_130= w_y_z_t;

% read grid
load hump_grid_nasa_les_coarse_noflow.dat


x1=hump_grid_nasa_les_coarse_noflow(:,1);
y1=hump_grid_nasa_les_coarse_noflow(:,2);

nim1_grid=x1(1);
njm1_grid=y1(1);

ni_grid=nim1_grid+1;
nj_grid=njm1_grid+1;


x=x1(2:end);
y=y1(2:end);

x2d=reshape(x,nim1_grid,njm1_grid);
y2d=reshape(y,nim1_grid,njm1_grid);

% compute cell centers
xp2d=zeros(ni_grid,nj_grid);
yp2d=zeros(ni_grid,nj_grid);

for jj=1:nj_grid
for ii=1:ni_grid

    im1=max(ii-1,1);
    jm1=max(jj-1,1);

    i=min(ii,nim1_grid);
    j=min(jj,njm1_grid);


   xp2d(ii,jj)=0.25*(x2d(i,j)+x2d(im1,j)+x2d(i,jm1)+x2d(im1,jm1));
   yp2d(ii,jj)=0.25*(y2d(i,j)+y2d(im1,j)+y2d(i,jm1)+y2d(im1,jm1));
end
end


dz=0.1/(nk-1);
disp(['dz= ',num2str(dz)])
z(1)=0;
for i=2:nk
   z(i)=z(i-1)+dz;
end

