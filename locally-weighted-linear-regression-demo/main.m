% �ݶ��½� - ���ݶ��½�
clc;
clear;
load data;
epsilon = 0.001; %������ֵ
alpha = 0.005; %ѧϰ��
tao = 0.5;    %����
testNum = size(test_feature, 1); %������������
n = size(X,2); %������+1
m = size(X,1); %ѵ����������
w = zeros(m,1); %Ȩֵ

for(i = 1:testNum)
    %���ݵ�ǰ�����������¼���Ȩֵ
    for(j = 1:m)
        w(j) = exp(-(X(j,:)-test_feature(i,:))*(X(j,:)-test_feature(i,:))'/(2*tao*tao));
    end;
    %������theta
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




