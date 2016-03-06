%X is n by dim
%Y is n by 1
%W is dim by 1
%b is 1 by 1
function [cost, dW, db] = svmGD(X,Y,W,b,C)
    [n, dim] = size(X);
    pred = X*W + b; %n by 1
    ypred = pred.*Y; %n by 1
    goodIndex = (ypred >= 1);%n by 1
    ycost = 1 - ypred;%n by 1
    ycost(goodIndex) = 0;%n by 1
    
    cost = W'*W*0.5 + C * sum(ycost);
    
    y = Y;%n by 1
    y(goodIndex) = 0;%n by 1
    
    yArr = repmat(y, 1, dim); %n by dim
    yx = yArr.*X; %n by dim
    yxSum = sum(yx,1); %1 by dim
    dW = W + C * (-yxSum'); %dim by 1
    
    db = -C* (sum(y));
end