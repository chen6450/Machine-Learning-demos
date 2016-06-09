load data;
X = [ones(size(train_feature,1),1) train_feature];
n = size(X,2); %特征数+1
m = size(X,1); %样本数
Y = train_label; %真实类别
W = zeros(n,1);  %增广的权重向量
WW = W;
alpha = 1;   %学习率
count = 0;  %连续正确分类的样本记数
converge = 0; %是否已经收敛
while converge == 0
	for i=1:m
        if Y(i)*(W'*X(i,:)') <= 0
            W = W + alpha*Y(i)*X(i,:)';
            WW = [WW W];
            count = 0;
        else
            count = count + 1;
            if count == m
                converge = 1;
                break;
            end;
        end;
    end;
end;
test_feature = [ones(size(test_feature,1),1) test_feature];
predict = sign(W'*test_feature')';