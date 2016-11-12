clear all
clf
warning off
format short

n_exp   = 5;
tol_grad = [1e-2 5*1e-3 1e-3 5*1e-4 1e-4 5*1e-5 1e-5];
Jtarget = 0.0468/2*ones(length(tol_grad));

erom       = zeros(length(tol_grad),n_exp);
Jrom       = zeros(length(tol_grad),n_exp);
bfgs_iters = zeros(length(tol_grad),n_exp);
trom       = zeros(length(tol_grad),n_exp);

%% Table 10
for i=1:n_exp
    [Jrom(:,i), bfgs_iters(:,i), trom(:,i)] = exper_BFGSperf(tol_grad,0);
end

% pick best run
for i=1:length(tol_grad)
   [trom_best,ind] = min(trom(i,:));
    
   disp(['BFGS tol_grad = ', num2str(tol_grad(i))])
   
   Jopt = Jrom(i,ind);
   num_iters = bfgs_iters(i,ind);
   walltime = trom(i,ind);
   disp(['$', num2str(Jopt), '$'])
   disp(['$', num2str(num_iters), '$'])
   disp(['$', num2str(walltime), '$'])
   
   disp(['----------------------------------------------'])
end