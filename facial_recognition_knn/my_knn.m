load faces.mat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  pre-computation of distances                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%for training data distance matrix
[tr_rows,tr_columns] = size(traindata);
train_distance = zeros(tr_rows,tr_rows);
for i = 1 : tr_rows
    for j = 1 : tr_rows
        if j < i
            train_distance(i,j) = train_distance(j,i);
            continue;
        end
        train_distance(i,j) = cosineDistance(traindata(i,:),traindata(j,:));       
    end    
end

%for testing data distance matrix
[te_rows,te_columns] = size(testdata);
test_distance = zeros(te_rows,tr_rows);
for i = 1 : te_rows
    for j = 1 : tr_rows
        test_distance(i,j) = cosineDistance(testdata(i,:),traindata(j,:));       
    end    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   KNN main algorithm start                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

k = [1 10 20 30 40 50 60 70 80 90 100];

training_error = zeros(1,length(k));
test_error = zeros(1,length(k));

for i = 1 : length(k)
    train_total=0;      %to hold the total number of training data class predictions
    train_correct=0;    %to hold the number of training correct class predictions
    test_total = 0;     %to hold the total number of test data class predictions
    test_correct=0;     %to hold the number of test correct class predictions
    
    %training error calculation
    for j = 1 : tr_rows
        [A,index] = sort(train_distance(j,:));
        k_nearest_indices = index(1:k(i));
        predicted_class = mode(trainlabels(k_nearest_indices));
        if predicted_class == trainlabels(j)
            train_correct = train_correct + 1;
        end
        train_total = train_total + 1;
    end
    
    %training error for each value of k
    training_error(i) = (train_total - train_correct) / train_total ;
    
    %test error calculation
    for j = 1 : te_rows
        [A,index] = sort(test_distance(j,:));
        k_nearest_indices = index(1:k(i));
        predicted_class = mode(trainlabels(k_nearest_indices));
        if predicted_class == testlabels(j)
            test_correct = test_correct + 1;
        end   
        test_total = test_total + 1;
    end   
    %test error for each value of k
    test_error(i) = (test_total - test_correct) / test_total ;
end    

%ploting the training and test error percentages
plot(k,training_error,'color','r','LineWidth',2); 
hold on;
xlabel('Value of k');
ylabel('Error');
title('Training and Test Error');
plot(k,test_error,'color','b','LineWidth',2);
hold off;
legend('Training error','Test error');   