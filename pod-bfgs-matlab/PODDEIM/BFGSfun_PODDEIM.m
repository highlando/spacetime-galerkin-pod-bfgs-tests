function [ f_red,g_red ] = BFGSfun_PODDEIM( u_red, z_red, y0_red )

global k3 Nt iter_tmp optOn Upod Uupod omega

U_red = reshape(u_red,k3,Nt);

Y_red = Burgers_Reduced(U_red,y0_red);
[~, g_red] = Adjoint_Grad_Reduced(Y_red, U_red, z_red);
f_red = costFun_Reduced(u_red,Y_red,z_red);

Urom = Uupod*U_red;
Yrom = Upod*Y_red; 

Yrom_bc  = [zeros(1,Nt);Yrom;zeros(1,Nt)];
Urom_bc  = [zeros(1,Nt);Urom;zeros(1,Nt)];
Jrom     = 0.5*L2norm(Yrom_bc,Zbc)^2 + omega/2*L2norm(Urom_bc,0*Urom_bc)^2;

% Force BFGS to stop

Jspacetime = [0.0702 0.0617 0.0468 0.0354 0.0304];
if Jrom <= Jspacetime(rr)/2
    g_red=0;
end



if optOn == 1
%     figure(2)
%     subplot(2,1,1)
%     mesh(time, space, [zeros(1,Nt);Upod*Y_red;zeros(1,Nt)]);
%     title('y(t,x)')
%     
%     subplot(2,1,2)
%     mesh(time, space, [zeros(1,Nt);Uupod*U_red;zeros(1,Nt)]);
%     title('u(t,x)')

    if iter == 1
        fprintf(1,['     #     f(x_k)    ||gradf(x_k)||  \n']);
    end
    
    fprintf('  %4d  %12.6e  %12.6e \n', ...
          iter_tmp, Jrom , norm(g_red));
    
end
iter_tmp = iter_tmp +1;

end

