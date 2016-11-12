function [z_red, y0_red] = setUp_redModel(Yss, Uss, z, y0)

global M B C k1 k3 Mred1 Mred2 Bred Cred Upod Uupod TMP

% SVD of snapshot matrices + DEIM
R = chol(M);
[Uy,~,~]=svd(R*Yss);
Upod=R\Uy(:,1:k1);

[Uupod,~,~]=svd(Uss);
Uupod = Uupod(:,1:k3);

% Pre-compute matrices
Mred1 = Upod'*M*Uupod;
Mred2 = Uupod'*M*Uupod;
Cred  = Upod'*C*Upod;
Bred  = Upod'*B;
TMP = zeros(k1,k1,k1);
for ii=1:k1
    for jj=1:ii
        TMP(:,ii,jj)=Bred*(Upod(:,ii).*Upod(:,jj));      
    end
end

z_red = Upod'*z;     % just because z is multiplied by y in J(.)
y0_red = Upod'*M*y0; % obtain reduced variables

end
