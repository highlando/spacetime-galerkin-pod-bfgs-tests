function [z_red, y0_red] = setUp_redModel(Yss, Uss, z, y0)

global M B C k1 k2 k3 Mred1 Mred2 Bred Cred Fred Upod Uupod TMP

% SVD of snapshot matrices + DEIM
R = chol(M);
[Uy,~,~]=svd(R*Yss);
Upod=R\Uy(:,1:k1);

[Uf,~,~]=svd(Yss.^2);
Udeim=Uf(:,1:k2);
[Pdeim,~]=deim(Udeim);

[Uupod,~,~]=svd(Uss);
Uupod = Uupod(:,1:k3);

% if plotOn ==1
%    figure(55)
%    index=1:min(N-1,Nt);
%    semilogy(index,diag(Sy(index,index)),'x',index,diag(Sf(index,index)),'rx',...
%        index,diag(Su(index,index)),'kx'); %investigate basis dimensions
% end

% Pre-compute matrices
Mred1 = Upod'*M*Uupod;
Mred2 = Uupod'*M*Uupod;
Bred  = Upod'*B*Udeim*inv(Pdeim'*Udeim);
Cred  = Upod'*C*Upod;
Fred  = Pdeim'*Upod;
TMP = zeros(k1,k1,k1);
for ii=1:k1
    for jj=1:ii
        TMP(:,ii,jj)=Bred*(Fred(:,ii).*Fred(:,jj));      
    end
end

z_red = Upod'*z;      % just because z is multiplied by y in J(.)
y0_red = Upod'*M*y0;  % obtain reduced variables
% U_red  = Uupod'*U;
% Y_red = Burgers_Reduced(U_red,y0_red);

end

