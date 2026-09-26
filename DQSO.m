
function [BestF, BestX, Convergence_curve] = DQSO_V9(nPop, MaxIt, Low, Up, Dim, fitness, tol, Display)
    if nargin < 8, Display = 'no'; end
    PopPos = zeros(nPop, Dim);
    PopFit = zeros(nPop, 1);
    for i=1:nPop   
        PopPos(i,:) = rand(1,Dim).*(Up-Low) + Low;
        PopFit(i) = feval(fitness, PopPos(i,:));   
    end
    [PopFit, indF] = sort(PopFit);
    PopPos = PopPos(indF, :);
    BestF = PopFit(1); BestX = PopPos(1, :);
    Convergence_curve = zeros(1, MaxIt);
    Momentum = zeros(nPop, Dim);
    space_diagonal = norm(Up - Low);
    if space_diagonal == 0, space_diagonal = 1e-10; end
    
    for It = 1:MaxIt
        % V9: Fractal Logarithmic Cooling (Parameter-Free)
        % Oscillates between 1 and 0 with exponentially doubling periods.
        % Perfectly replaces (1 - It/MaxIt)^3 without knowing MaxIt!
        if It == 1
            T = 1.0;
        else
            T = (cos(pi/2 * log2(It)))^2;
        end
        
        mean_dist = 0;
        for i = 1:nPop
            mean_dist = mean_dist + norm(PopPos(i,:) - BestX);
        end
        mean_dist = mean_dist / nPop;
        C_index_raw = mean_dist / space_diagonal; 
        
        C_index = sqrt(C_index_raw) * T; 
        if C_index < 1e-4 * T
            C_index = 1e-4 * T;
        end
        
        newPopPos = zeros(nPop, Dim);
        newPopPos(nPop,:) = BestX + randn(1, Dim) .* (Up - Low) * C_index;
        
        j_idx = zeros(nPop-1, 1);
        for i=1:nPop-1, j_idx(i) = randi([i+1, nPop]); end
        rank_prob_2 = 1 - (1:nPop-1)' / nPop;
        mask_sigma = rand(nPop-1, 1) < rank_prob_2;
        r1 = rand(nPop-1, 1); r2 = rand(nPop-1, 1);
        R1 = rand(nPop-1, Dim); R2 = rand(nPop-1, Dim);
        M1 = r1 .* mask_sigma + R1 .* (~mask_sigma);
        M2 = r2 .* mask_sigma + R2 .* (~mask_sigma);
        
        newPopPos(1:nPop-1, :) = PopPos(1:nPop-1, :) + M1 .* (PopPos(1:nPop-1, :) - PopPos(j_idx, :)) + M2 .* (repmat(BestX, nPop-1, 1) - PopPos(1:nPop-1, :));
        newPopPos = SpaceBound(newPopPos, Up, Low);
        
        for i = 1:nPop
            fnew = feval(fitness, newPopPos(i,:));
            if fnew < PopFit(i)
                PopFit(i) = fnew; PopPos(i,:) = newPopPos(i,:);
            end
        end
        [PopFit, indF] = sort(PopFit);
        PopPos = PopPos(indF, :); BestX = PopPos(1, :); BestF = PopFit(1);

        rank_prob_3 = 1 - (1:nPop)' / nPop;
        mask_sigma3 = rand(nPop, 1) < rank_prob_3;
        mask_up = rand(nPop, 1) < 0.5; mask_1d = rand(nPop, 1) < 0.5;
        radius = repmat(Up - BestX, nPop, 1) .* mask_up + repmat(BestX - Low, nPop, 1) .* (~mask_up);
        R_1d = randn(nPop, 1); R_nd = randn(nPop, Dim);
        step_boundary = 3 * (R_1d .* mask_1d + R_nd .* (~mask_1d)) .* radius * C_index;
        step_sigma = R_1d .* (repmat(BestX, nPop, 1) - PopPos) * C_index;
        step = step_sigma .* mask_sigma3 + step_boundary .* (~mask_sigma3);
        
        Momentum = (1 - C_index) * Momentum + C_index * step;
        Momentum = Momentum .* T^2; 
        
        newPopPos = repmat(BestX, nPop, 1) + step + Momentum;
        newPopPos = SpaceBound(newPopPos, Up, Low);
        
        for i = 1:nPop
            fnew = feval(fitness, newPopPos(i,:));
            if fnew < PopFit(i)
                PopFit(i) = fnew; PopPos(i,:) = newPopPos(i,:);
            end
        end
        [PopFit, indF] = sort(PopFit);
        PopPos = PopPos(indF, :); BestX = PopPos(1,:); BestF = PopFit(1);
        
        if T < 0.2
            singularity_pos = rand() * BestX;
            singularity_pos = max(singularity_pos, Low); singularity_pos = min(singularity_pos, Up);
            f_sing = feval(fitness, singularity_pos);
            if f_sing < BestF
                BestX = singularity_pos; BestF = f_sing;
                PopPos(nPop,:) = BestX; PopFit(nPop) = BestF;
            end
        end
        
        Convergence_curve(It) = BestF;        
    end 
end
function  X = SpaceBound(X, Up, Low)
    n = size(X, 1); UpMat = repmat(Up, n, 1); LowMat = repmat(Low, n, 1);
    Flag4ub = X > UpMat; Flag4lb = X < LowMat;
    X = (X .* (~(Flag4ub + Flag4lb))) + UpMat .* Flag4ub + LowMat .* Flag4lb;
end

