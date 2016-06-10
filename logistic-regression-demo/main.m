load data;
X = [ones(size(train_feature,1),1) train_feature];
n = size(X,2); %特征数+1
m = size(X,1); %样本数
Y = train_label; %真实类别
theta = zeros(n,1); %参数
theta_new = zeros(n,1);
epsilon = 0.0001; %收敛阈值
alpha = 0.01; %学习率

k = 1;
figure(1);
while 1
    % 计算参数的似然性(仅用作分析)
	L(k) = 0;
    for(i = 1:m)
        h = 1/(1+exp(-theta'*X(i, :)'));
        L(k) = L(k) + (Y(i)*log(h) + (1-Y(i))*log(1-h));
    end;
    L(k) = exp(L(k));
    fprintf('The %dth iteration, L = %f \n', k, exp(L(k)));
    for(j = 1:n)
        theta_new(j) = theta(j);
        for(i = 1:m)
            h = 1/(1+exp(-theta'*X(i, :)'));
            theta_new(j) = theta_new(j) + alpha * (Y(i)-h) * X(i, j); 
        end;
    end;
	if norm(theta_new-theta) < epsilon
        theta = theta_new;
		break;
    else
        theta = theta_new;
        k = k + 1;
	end;
end;

% 测试
test_feature = [ones(size(test_feature,1),1) test_feature];
predict=[];
predictLabel=[];
for i=1:size(test_feature,1)
    predict=[predict;1/(1+exp(-theta'*test_feature(i,:)'))];
    if 1/(1+exp(-theta'*test_feature(i,:)'))>=0.5
        predictLabel=[predictLabel;1];
    else
        predictLabel=[predictLabel;0];
    end;
end;

% 画图
plot(L);

