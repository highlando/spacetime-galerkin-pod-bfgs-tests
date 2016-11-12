function [Lambda, gradf] = Adjoint_Grad(Yfull, U, z)


global N Ns T M C nu h omega


dt= T/(Ns-1); 

%% Compute the adjoint Lambda
Lambda = zeros(N-1,Ns); 
% terminal condition
Mat = (1/dt)*M + Ny(Yfull(:,end)) + nu*C ;
rhs = -dt*( M*Yfull(:,end) - h*z );
Lambda(:,end) = Mat'\rhs;

% Solving backwards
for ii=(Ns-1):-1:1
    Mat = (1/dt)*M + Ny(Yfull(:,ii)) + nu*C;
    rhs = -( (-(1/dt)*M)'*Lambda(:,ii+1) ) - dt*( M*Yfull(:,ii) - h*z );
    
    Lambda(:,ii) = Mat'\rhs;
end


%% Compute gradf
Gradf = zeros(N-1,Ns); 
Gradf(:,1) = dt*omega*M*U(:,1);
for ii=2:Ns
    Gradf(:,ii) = dt*omega*M*U(:,ii) - M'*Lambda(:,ii);
end

gradf  = reshape(Gradf,(N-1)*Ns,1);

end