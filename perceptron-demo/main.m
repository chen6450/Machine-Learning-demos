load data;
X = [ones(size(train_feature,1),1) train_feature];
n = size(X,2); %������+1
m = size(X,1); %������
Y = train_label; %��ʵ���
W = zeros(n,1);  %�����Ȩ������
WW = W;
alpha = 1;   %ѧϰ��
count = 0;  %������ȷ�������������
converge = 0; %�Ƿ��Ѿ�����
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