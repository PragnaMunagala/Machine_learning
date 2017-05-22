%%%reading input data
M = csvread('C:/Users/munag/Desktop/ASU/2nd Sem Courses Files/SML/Assignments/Assignment2/breast-cancer-wisconsin.dat');
X = M(:,2:10);
y = M(:,11);
%%%%replacing data class labels 2 with 1 and 4 with -1
for i = 1 : length(y)
    if y(i) == 2
        y(i) = 1;
    else
        y(i) = -1;
    end    
end  
%%%%Naive Bayes Classifier
[train_size_array_n,acc_n] = Naive_Bayes(X, y);
%%%plotting accuracy bar graph for naive bayes
bar(log(train_size_array_n),acc_n,'red');

%%%reading input data
M = csvread('C:/Users/munag/Desktop/ASU/2nd Sem Courses Files/SML/Assignments/Assignment2/breast-cancer-wisconsin.dat');
X = M(:,2:10);
y = M(:,11);
%%%%replacing data class labels 2 with 1 and 4 with 0
for i = 1 : length(y)
    if y(i) == 2
        y(i) = 1;
    else
        y(i) = 0;
    end    
end 
%%%logistic regression classifier
[train_size_array,acc] = Logistic_Regression(X, y);
hold on;
%%%plotting accuracy bar graph for logistic regression
bar(log(train_size_array),acc,'blue');
xlabel('Log of size of training data set');
ylabel('Accuracy');
title('Learning curve');
legend('Naive Bayes','Logistic Regression')