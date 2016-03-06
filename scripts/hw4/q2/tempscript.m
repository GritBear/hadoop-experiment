
%batch
learningRate = 0.0000003;
thresh = 0.25;

b = 0;
W = zeros(dim,1);

batchCost = [];
batchDeltaCost = [];
iter = 0;
errorDelta = 100000;
while(errorDelta > thresh && iter < 20000)
    [cost, deltaW, deltab] = svmGD(X_all, Y_all, W, b, C);
    
    batchCost(end + 1) = cost;
    
    W = W - learningRate * deltaW;
    b = b - learningRate * deltab;
    
    if(iter > 0)
        errorDelta = abs(cost - batchCost(iter))* 100/batchCost(iter) ;
        batchDeltaCost(end + 1) = errorDelta;
    end
    
    iter = iter + 1;
end

batchIter = iter