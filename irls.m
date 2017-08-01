function [x, xks, residuals, executed_iters] = irls(A, b, opts)
%
% Brief: Iteratively Reweighted Least Squares (IRLS) proposed in the paper 
%        "Iteratively reweighted algorithms for compressive sensing", 
%         by Rick Chartrand and Wotao Yin
%
% Inputs:  x               - 1d vector. an input signal 
%
%          opts  
%              opts.max_iters   - maximum iterations
%
%              opts.p           - lp in [0,1]
%
%              opts.x0          - initial point given by users. If not 
%                                 defined, we will calculate and use the LS solution   
%
%              opts.epsilon_sequence - 'exponential_decay_base_2', 'rcwy', 'none'
%
%              opts.x_truth     - the groundtruth 
%
%              opts.stop_eps    - stop the program when the solution reaches the x_truth with the presion stop_eps
%
%              opts.verbose     -  true or false or a number 
%
%              opts.l2_solver   - 'direct', or 'pcg' or 'lsqlin'
%
% Outputs: x         - transformed signal in the frequency domain  
%          xks       - all intermediate x
%          residuals - residuals
%
%
% Written by: Jinchao Liu 
% Email: liujinchao2000@gmail.com
% Created: Nov 2015
% Modified: Dec 2015
% Version: 0.1


[m, n] = size(A);
if nargin < 3 opts = []; end

if ~isfield(opts, 'max_iters') opts.max_iters = 1e3; end
if ~isfield(opts, 'p'        ) opts.p = 1; warning('I did not see a valid options.p, use p = 1.'); end
if ~isfield(opts, 'epsilon_sequence'  ) opts.epsilon_sequence = 'exponential_decay_base_2'; end
if ~isfield(opts, 'verbose'  ) opts.verbose = false; end
if ~isfield(opts, 'x0'       ) opts.x0 = []; end
if ~isfield(opts, 'x_truth'  ) opts.x_truth = []; end
if ~isfield(opts, 'stop_eps' ) opts.stop_eps = []; end

p = opts.p;
verbose = opts.verbose;
max_iters = opts.max_iters;
epsilon_sequence = opts.epsilon_sequence;

epsilon = 1; epsilonf = 1e-8;
if strcmp(epsilon_sequence,'none') epsilon = epsilonf; end

% Initial solution 
if isempty(opts.x0) x = A'*inv(A*A')*b; % initialize to the least-l2-norm solution 
else                x = opts.x0;
end
xks = [x];
w = ones(n,1)./n;

% Main body
residuals = [];
epsilons = [];
executed_iters = 0;

for i = 1 : max_iters     
    
    % Slove a weighted constrained least square problem
    if strcmp(opts.l2_solver, 'pcg')
        Qn = diag((xks(:,end).^2 + epsilon).^(1-p*0.5));
        [xg, flag] = pcg(A*Qn*A',b,1e-6,100);
        x = Qn*A'*xg;
    elseif strcmp(opts.l2_solver, 'lsqlin') 
        C = sqrt(diag((xks(:,end).^2 + epsilon).^(p*0.5-1)));
        d = zeros(size(C,1),1);
        options = optimset('Algorithm','trust-region-reflective','Display','off','LargeScale','off');
        x = lsqlin(C, d, [], [], A, b, [], [], [], options);    
    else
        if ~strcmp(opts.l2_solver, 'direct')
            warning('opts.l2_solver is not set properly, use default.');
        end
        Qn = diag((xks(:,end).^2 + epsilon).^(1-p*0.5));
        x = Qn*A'*((A*Qn*A')\b);
    end   

    % Generate epsilon sequence         
    if strcmp(epsilon_sequence,'none')
        epsilon = epsilonf;
    end
    if strcmp(epsilon_sequence,'rcwy')
        if abs(norm(x) - norm(xks(:,end))) < sqrt(epsilon)/100
            epsilon = epsilon * 0.1; continue;
        end  
        if epsilon<epsilonf break; end % Exit if epsilon is small enough
    end    
    if strcmp(epsilon_sequence,'exponential_decay_base_2')
        epsilon = 1/2.^(i-1);
        if epsilon < epsilonf epsilon = epsilonf; end 
    end  
    
    % Store some intermediate results for next iteration
    residuals = [residuals, norm(b - A*x)];
    xks = [xks, x];           
    
    % Verbose
    if rem(i, verbose) == 0 fprintf('Iter: %i  Residual: %d \n', i, residuals(end)); end        
   
    % Early stop
    if ~isempty(opts.stop_eps) && ~isempty(opts.x_truth)
        if norm(x-opts.x_truth) < opts.stop_eps break; end
    end    
end

executed_iters = i;
