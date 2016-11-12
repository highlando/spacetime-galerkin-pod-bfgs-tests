close all
clear all
clf 

addpath('./FOM');
addpath('./POD');
% addpath('./PODDEIM');

global nu tol_newton max_newton h dt omega M B C Nt N k1 k2 k3...
    time space plotOn Uupod Upod iter_tmp


flag = 2;

if flag == 1 
    addpath('./SPG');
elseif flag == 2
    addpath('./BFGS','./BFGS/immoptibox/');
elseif flag == 3
    addpath('./Newton-type');
end


%% Params

nu    = 0.005;           % viscosity parameter
omega = 10e-2;           % control penalty

N  = 219;               % # grid point in space
Nt = 120;                % # grid point in time
L  = 1;                 % x \in [0,L]
T  = 1;                 % t \in [0,T]

x = linspace(0,L,N+1);           % Spacial discretization
t = linspace(0,T,Nt);            % time discretization
[time, space] = meshgrid(t',x);  % for 3D plotting
h = L/N;                         % constant step size
dt= T/(Nt-1);                    % constant time step

tol_newton = 10e-3;     % for Newton's method
max_newton = 20;        % max. number of inner Newton iterations

k1 = 24;                 % # POD basis of y
k2 = 15;                 % # DEIM basis
k3 = k1;                 % # POD basis of u

plotOn = 0;             % plotting and output on (1) or off(0)


%% Initialize the state y_0
y0 = [ones(floor((N-1)/2),1);zeros(ceil((N-1)/2),1)];

%% desired state z, does not depend on time
z  = y0;

%% Pre-compute stiffness and mass matrix for full-order Burgers eqn
M = (h/6)*gallery('tridiag',ones(N-2,1),4*ones(N-1,1),ones(N-2,1));
B = gallery('tridiag',-0.5*ones(N-2,1),zeros(N-1,1),0.5*ones(N-2,1));
C = (1/h)*gallery('tridiag',-ones(N-2,1),2*ones(N-1,1),-ones(N-2,1));

%% Initialize the control u_0 = 0
u0 = zeros((N-1)*Nt,1);


%% Optimization - full order

% ... using SPG method
if flag == 1
    Opts = struct('x0',u0,'n',(N-1)*Nt);
     xl = -2;
     xr = 2;
     Opts.Proj = @(x)min(max(x,xl),xr); 

    tStart = tic;
    [u,INFO,u_ss] = spg_cons(@(u)SPGfun(u, z, y0), Opts);
    tElapsed = toc(tStart)
end

% ... using BFGS method
if flag == 2
    Opts = [1 1e-8 10e-3 200];
    iter_tmp = 0;
    
    tStart = tic;
    u = ucminf(@(u)BFGSfun(u, z, y0), u0,Opts);
    tElapsed = toc(tStart)
    
    load ./BFGS/snapshots.mat
end

% ...using Newton-eqn + adjoints
if flag == 3
    Opts = struct('max_opt',30,'tol_opt_f',10e-8,'tol_opt_gradf',10e-9,'tol_line',10e-4,'min_alpha',10e-8,'eta',10e-2);

    tStart = tic;
    [u,u_ss] = minNewtonCG(u0,y0,z,Opts);
    tElapsed = toc(tStart)
end

Yfinal = Burgers(reshape(u,N-1,Nt),y0);

%% Optimization - reduced model

u_ss = 0*u_ss;
U = reshape(u_ss,N-1,Nt);
Yapprox = Burgers(U,y0);

tStart = tic;
[~, z_red, y0_red] = setUp_redModel(Yapprox, U, z, y0);
tElapsed_Setup = toc(tStart)

% ... using SPG
if flag == 1
    Opts = struct('x0',u0_red,'n',k3*Nt);
    xl = -2;
    xr = 2;
    Uu_big = kron(eye(Nt),Uupod);
    Opts.Proj = @(x) Uu_big'*min(max(Uu_big*x,xl),xr); 
    tStart = tic;
    u_red = spg_cons(@(u_red)SPGfun_Reduced(u_red, z_red, y0_red), Opts);
    tElapsed_red = toc(tStart)
end

% ... using BFGS
if flag == 2
    iter_tmp = 0;

    tStart = tic;
    u_red = ucminf(@(u_red)BFGSfun_Reduced(u_red, z_red, y0_red), u0,Opts);
    tElapsed_red = toc(tStart)
end

% ... using Newton-eqn + adjoints
if flag == 3
    tStart = tic;
    u_red = minNewtonCG_Reduced(u0,y0_red,z_red,Opts);
    tElapsed_red = toc(tStart)
end



U = reshape(u_red,N-1,Nt);
Y_red = Burgers_Reduced(reshape(u_red,N-1,Nt),y0_red);
Yapprox = Upod*Y_red;


%% Some numerical analysis

SpeedUp = tElapsed/tElapsed_red

Z = kron(ones(1,Nt),z);
% error     = L2norm(Yfinal,Z)
% error_red = L2norm(Yapprox,Z)
% error = L2norm(Yfinal,Yapprox)


Ybc  = [zeros(1,Nt);Yapprox;zeros(1,Nt)];
Zbc  = [zeros(1,Nt);Z;zeros(1,Nt)];
Ubc =  [zeros(1,Nt);U;zeros(1,Nt)];

cost_fun = 0.5*L2norm(Ybc,Zbc)^2 + omega/2*L2norm(Ubc,0*Ubc)^2
error    = L2norm(Ybc,Zbc)
        

