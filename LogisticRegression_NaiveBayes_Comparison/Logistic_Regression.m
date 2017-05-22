function [train_size_array,acc] = Logistic_Regression(X,y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%random fractions of training data
    random_frac = [.01 .02 .03 .125 .625 1];
    acc = zeros(length(random_frac),1);   %accuracy values array
    train_size_array = zeros(length(random_frac),1);   %size of training data for different random fractions
    
    %%%%for runs on training data
    for mainloop = 1 : length(random_frac)
        
        %%%training data size and X and y vectors
        train_size = round(random_frac(mainloop)* 2 * length(y) / 3);  
        train_size_array(mainloop) = train_size;
        train_X = X(1:train_size,:);
        train_X = [ones(train_size,1) train_X];
        train_y = y(1:train_size);
        
        %%test data X and y vectors
        test_y = y(train_size+1:length(y));
        test_X = X(train_size+1:length(y),:);
        test_X = [ones(length(test_y),1) test_X];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %training classifier%
        [rows,columns] = size(train_X);
        initial_weights = zeros(columns, 1);
        options = optimset('GradObj', 'on', 'MaxIter', 1000);
        
        %%%calculating the minimum
        [weight, cost] = fminunc(@(t)(cost_function(t, train_X, train_y)), initial_weights, options);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %testing classifier%
        p = 1 ./ (1 + exp(-1 * test_X * weight));
        for i=1:length(test_y)
            if and(p(i) >= 0.5, test_y(i)==1)
                acc(mainloop) = acc(mainloop) + 1;
            elseif and(p(i) < 0.5, test_y(i)==0)
                acc(mainloop) = acc(mainloop) + 1;
            end    
        end
        acc(mainloop) = acc(mainloop) / length(test_X); %%%accuracy value
    end
accmean = mean(acc(1:5));   %%%average of accuracies  
disp(accmean);
end