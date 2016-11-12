function [Jrom, bfgs_iters, trom] =  exper_BFGSperf(tol_grad, DEIM_flag)

close all
% clear all
% clf 

addpath('./FOM');
addpath('./BFGS','./BFGS/immoptibox/');

global nu tol_newton max_newton h omega M B C Ns Nt N T L k1 k2 k3 ...
    time space plotOn optOn Uupod Upod iter_tmp Zbc u_ss


% tol_grad = [1e-2 5*1e-3 1e-3 5*1e-4 1e-4 5*1e-5 1e-5];
n_exp =length(tol_grad);
qhat = 18*ones(n_exp);
nt   = 18*ones(n_exp);
Jtarget = 0*ones(n_exp);

%% Params

nu    = 0.005;        % viscosity parameter
omega = 0.001;        % control penalty

N  = 219;              % # grid point in space
Ns = 120;              % # snapshots

L  = 1;                % x \in [0,L]
T  = 1;                % t \in [0,T]


h = L/N;                         % constant step size
% dt= T/(Nt-1);                    % constant time step

tol_newton = 10e-3;     % for Newton's method
max_newton = 20;        % max. number of inner Newton iterations


%% Target and init cond
y0  = [ones(floor((N-1)/2),1);zeros(ceil((N-1)/2),1)];
z   = y0;
Z   = kron(ones(1,Ns),z);
Zbc = [zeros(1,Ns);Z;zeros(1,Ns)];
    
%% Pre-compute stiffness and mass matrix for full-order Burgers eqn
M = (h/6)*gallery('tridiag',ones(N-2,1),4*ones(N-1,1),ones(N-2,1));
B = gallery('tridiag',-0.5*ones(N-2,1),zeros(N-1,1),0.5*ones(N-2,1));
C = (1/h)*gallery('tridiag',-ones(N-2,1),2*ones(N-1,1),-ones(N-2,1));

%% FOM optimization

%Solve FOM optimization with BFGS
tStart = tic;
iter_tmp = 0;
Opts = [1 1e-8 1e-6 1];
optOn = 0;
u0 = zeros((N-1)*Ns,1);
ufom = ucminf(@(u)BFGSfun(u, z, y0), u0,Opts);
tElapsed = toc(tStart);

Ufom = reshape(ufom,N-1,Ns);
Yfom = Burgers(Ufom,y0);


plotOn = 0; 
optOn  = 0;


%% ROM optimization
% load ./BFGS/snapshots.mat
if DEIM_flag==0
    addpath('./POD');
else
    addpath('./PODDEIM');
end

Jrom         = zeros(n_exp,1);
bfgs_iters   = zeros(n_exp,1);
trom         = zeros(n_exp,1);

for rr = 1:n_exp
    % specify reduced dimensions
    k1 = qhat(rr);    % # POD basis of y
    k2 = 25;          % # DEIM basis
    k3 = qhat(rr);    % # POD basis of u

    Nt = nt(rr);    % # grid point in time for red model
  
    tStart = tic;
    Uss = reshape(u_ss,N-1,Ns);
    Yss = Burgers(Uss,y0);
    [z_red, y0_red] = setUp_redModel(Yss, Uss, z, y0);
    tElapsed_Setup = toc(tStart);

    % Solve ROM optimization with BFGS
    tStart = tic;
    iter_tmp = 0;
    Opts = [1 1e-16 tol_grad(rr) 400];
    u0_red = zeros(k3*Nt,1); 
    [u_red, INFO] = ucminf(@(u_red)BFGSfun_Reduced(u_red, z_red, y0_red, Jtarget, rr), u0_red,Opts);
    tElapsed_red = toc(tStart);
    
    Y_red = Burgers_Reduced(reshape(u_red,k3,Nt),y0_red);
     
    Urom = Uupod*reshape(u_red,k3,Nt);
    Yrom = Upod*Y_red;    
    
    
    
    %% Analysis
    % SpeedUp  = tElapsed/tElapsed_red
%     Yfom_bc  = [zeros(1,Nt);Yfom;zeros(1,Nt)];
%     Ufom_bc  = [zeros(1,Nt);Ufom;zeros(1,Nt)];
    Yrom_bc  = [zeros(1,Nt);Yrom;zeros(1,Nt)];
    Urom_bc  = [zeros(1,Nt);Urom;zeros(1,Nt)];
    
%     efom    = L2norm(Yfom_bc,Zbc)^2;
%     Jfom    = 0.5*L2norm(Yfom_bc,Zbc)^2 + omega/2*L2norm(Ufom_bc,0*Ufom_bc)^2;
    erom    = L2norm(Yrom_bc,Zbc)^2;
    Jrom(rr)    = 0.5*L2norm(Yrom_bc,Zbc)^2 + omega/2*L2norm(Urom_bc,0*Urom_bc)^2;
    bfgs_iters(rr) = iter_tmp;
    trom(rr) = tElapsed_red;
%     disp(['BFGS tol_grad = ', num2str(tol_grad(rr))])
%     erom
%     Jrom
%     bfgs_iters = iter_tmp
%     tElapsed_red
%     disp(['----------------------------------------------'])
    
    if plotOn == 1
%         Yu0 = Burgers(0*Ufom,y0);
%         figure(1)
%         mesh(time, space, [zeros(1,Nt);Yu0;zeros(1,Nt)]);
%         title('FOM - uncontroled state')
% 
%         figure(2)
%         mesh(time, space, Yfom_bc);
%         title('FOM - optimal state')
% 
%         figure(3)
%         mesh(time, space, Ufom_bc);
%         title('FOM - optimal control')


        x = linspace(0,L,N+1);           % Spacial discretization
        t = linspace(0,T,Nt);            % time discretization
        [time, space] = meshgrid(t',x);  % for 3D plotting of reduced model

        figure(100+rr)
        mesh(time, space, Yrom_bc);
        title('ROM - optimal state')

        figure(200+rr)
        mesh(time, space, Urom_bc);
        title('ROM - optimal control')
    end
end     

rmpath('./POD');
rmpath('./PODDEIM');