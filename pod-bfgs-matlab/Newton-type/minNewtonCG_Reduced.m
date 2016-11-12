function u_red = minNewtonCG_Reduced(u_red,y0_red,z_red,Opts)

global tol_line min_alpha eta k3 Nt time space Upod Uupod plotOn

max_opt       = Opts.max_opt;        % optimization iterations     

tol_opt_f     = Opts.tol_opt_f;      % change in objective function 
tol_opt_gradf = Opts.tol_opt_gradf;  % tolerance for gradient

tol_line = Opts.tol_line;            % tolerance for line search
min_alpha = Opts.min_alpha;          % minimum step size in direction s_k

eta = Opts.eta;                      % tolerance for truncated CG


U_red = reshape(u_red,k3,Nt);
Y_red  = Burgers_Reduced(U_red,y0_red);

for k = 1:max_opt
    tic;
    
    cost_old = costFun_Reduced(u_red,Y_red,z_red);

    %% Compute grad_f via adjoint I 
    [Lambda_red, gradf_red] = Adjoint_Grad_Reduced(Y_red, U_red, z_red);

    % Check for stopping of optimization
    if (norm(gradf_red) < tol_opt_gradf) % first order optimality condition
        disp('Zero-gradient criterium satisfied.');
        return
    end    

    %% Solve Newton eqn. for s_k via CG alg, Hess_f*v is calculated via adjoint II
    s0_red = zeros(k3*Nt,1);
    [s_red,iter] = TruncatedCG( @(u_red)Adjoint_HessVec_Reduced(u_red,Y_red,Lambda_red), s0_red, -gradf_red, k3*Nt );

    %% Perform line search in s_k direction    
    [alpha, Y_red, u_red] = ArmijoLineSearch_Reduced( @costFun_Reduced, Y_red, y0_red, z_red, u_red, s_red, s_red'*gradf_red );    
    U_red = reshape(u_red,k3,Nt); % update
    
    
    cost_new=costFun_Reduced(u_red,Y_red,z_red); 

    if (abs(cost_new-cost_old) < tol_opt_f)
        disp('No further improvement in objective function');
        return
    end
    
    count = toc;
    
    if plotOn == 1
        PlotAndOutput_Reduced(time,space, Upod*Y_red, Uupod*U_red, ...
            Y_red, u_red, z_red, k, gradf_red, s_red, alpha, count, iter);
    end
end




end


