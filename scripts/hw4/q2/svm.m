clear all

X_all = dlmread('features.txt',',');
Y_all = dlmread('target.txt');
[n, dim] = size(X_all);
C = 100;

%SGD
learningRate = 0.0001;
thresh = 0.001;

order = randperm(n);

b = 0;
W = zeros(dim,1);

SGDCost = [];
SGDDeltaCost = [];
iter = 0;
errorDelta = 100000;
while(errorDelta > thresh && iter < 20000)
    sampleInd = mod(iter, n) + 1;
    selection = order(sampleInd);
    
    [cost, deltaW, deltab] = svmGD(X_all(selection,:), Y_all(selection), W, b, C);

    W = W - learningRate * deltaW;
    b = b - learningRate * deltab;
    
    if(iter == 1)
        errorDelta = abs(cost - SGDCost(end))* 100/SGDCost(end) * 0.5;
        SGDDeltaCost(end + 1) = errorDelta;
    elseif(iter >=1)
        temp = abs(cost - SGDCost(end))* 100/SGDCost(end) * 0.5;
        errorDelta = errorDelta * 0.5 + temp * 0.5;
        SGDDeltaCost(end + 1) = errorDelta;
    end
    SGDCost(end + 1) = cost;
    iter = iter + 1;
end

SGDIter = iter
plot(SGDCost)
