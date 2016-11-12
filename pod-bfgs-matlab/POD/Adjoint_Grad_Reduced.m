function [Lambda_red, gradf_red] = Adjoint_Grad_Reduced(Y_red, U_red, z_red)


global k1 k3 Nt Cred Mred1 Mred2 T nu h omega

dt= T/(Nt-1); 

%% Compute the adjoint Lambda
Lambda_red = zeros(k1,Nt); 
% terminal condition
Mat = (1/dt)*eye(k1) + Ntildey(Y_red(:,end)) + nu*Cred ;
rhs = -dt*( Y_red(:,end) - h*z_red );
Lambda_red(:,end) = Mat'\rhs;

% Solving backwards
for ii=(Nt-1):-1:1
    Mat = (1/dt)*eye(k1) + Ntildey(Y_red(:,ii)) + nu*Cred;
    rhs = -( (-(1/dt)*eye(k1))'*Lambda_red(:,ii+1) ) - dt*( Y_red(:,ii) - h*z_red );
    
    Lambda_red(:,ii) = Mat'\rhs;
end

%% Compute gradf
Gradf_red      = zeros(k3,Nt);
Gradf_red(:,1) = dt*omega*Mred2*U_red(:,1); 
for ii=2:Nt
    Gradf_red(:,ii) = dt*omega*Mred2*U_red(:,ii) - Mred1'*Lambda_red(:,ii);
end


gradf_red  = reshape(Gradf_red,k3*Nt,1);
end

