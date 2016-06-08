% 梯度下降 - 批梯度下降
clc;
clear;
load data;
epsilon = 0.001; %收敛阈值
alpha = 0.005; %学习率
tao = 0.5;    %波长
testNum = size(test_feature, 1); %测试样本个数
n = size(X,2); %特征数+1
m = size(X,1); %训练样本个数
w = zeros(m,1); %权值

for(i = 1:testNum)
    %根据当前测试样本重新计算权值
    for(j = 1:m)
        w(j) = exp(-(X(j,:)-test_feature(i,:))*(X(j,:)-test_feature(i,:))'/(2*tao*tao));
    end;
    %求解参数theta
    theta = zeros(n,1);
    theta_new = zeros(n,1);
    while 1
        for(j = 1:n)
            theta_new(j) = theta(j);
            for(k = 1:m)
                theta_new(j) = theta_new(j) - alpha * w(k) * (X(k, :) * theta - Y(k, :))*X(k, j);
            end;
        end;
        if norm(theta_new-theta) < epsilon
            break;
        end;
        theta = theta_new;
    end;
    predict(i) = test_feature(i,:)*theta;
end;




