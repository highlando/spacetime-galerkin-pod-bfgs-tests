function hessvec_red = Adjoint_HessVec_Reduced( v_red, Y_red, Lambda_red )


global k1 k3 Nt Cred Mred1 Mred2 nu dt omega

V_red = reshape(v_red,k3,Nt);

%% Solve for w
w=zeros(k1,Nt);
w(:,1) = zeros(k1,1); %initial condition
for ii = 2:Nt
    Mat = (1/dt)*eye(k1) + Ntildey(Y_red(:,ii)) + nu*Cred;
    rhs = -(-(1/dt)*eye(k1))*w(:,ii-1) - Mred1*V_red(:,ii);
    
    w(:,ii) = Mat\rhs;
end

%% Solve for p
p=zeros(k1,Nt);
Mat = (1/dt)*eye(k1) + Ntildey(Y_red(:,end)) + nu*Cred;
rhs = ( dt*eye(k1) + Ntildeyy(Lambda_red(:,end)) )*w(:,end);

p(:,end) = Mat'\rhs; %terminal condition

for ii = (Nt-1):-1:1
    Mat = (1/dt)*eye(k1) + Ntildey(Y_red(:,ii)) + nu*Cred;
    rhs = -( -((1/dt)*eye(k1)) )'*p(:,ii+1) + ( dt*eye(k1) + Ntildeyy(Lambda_red(:,ii)) )*w(:,ii);
    p(:,ii) = Mat'\rhs;
end

%% Calculate Hess_f*u
HessVec_red      = zeros(k3,Nt);
HessVec_red(:,1) = dt*omega*Mred2*V_red(:,1);
for ii = 2:Nt
    HessVec_red(:,ii) = -Mred1'*p(:,ii) + (dt*omega*Mred2)*V_red(:,ii);
end


hessvec_red = reshape(HessVec_red,k3*Nt,1);
end

