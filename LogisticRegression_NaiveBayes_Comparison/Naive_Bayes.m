function [train_size_array,acc] = Naive_Bayes(X, y)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%random fractions of training data
    random_frac = [.01 .02 .03 .125 .625 1];
    acc = zeros(length(random_frac),1);    %accuracy values array
    train_size_array = zeros(length(random_frac),1);        %size of training data for different random fractions
    
    %%%%for runs on training data
    for mainloop = 1 : length(random_frac)
        
        %%%training data size and X and y vectors
        train_size = round(random_frac(mainloop)* 2 * length(y) / 3);  
        train_size_array(mainloop) = train_size;
        train_X = X(1:train_size,:);
        train_y = y(1:train_size);
        
        %%test data X and y vectors
        test_X = X(ceil(2 * length(y) / 3 + 1) : length(y),:);
        test_y = y(ceil(2 * length(y) / 3 + 1) : length(y));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%training classifier%%
        prob_pos = 0;
        prob_neg = 0;
        pos_count = 0;
        neg_count = 0;
        
        %%calculating the count of positive and negative class labels
        for i = 1 : length(train_y)
            if train_y(i) == 1
                pos_count = pos_count + 1; 
            else
                neg_count = neg_count + 1;     
            end    
        end
        
        %%probability of positive and negative class labels
        prob_pos = pos_count / train_size;
        prob_neg = neg_count / train_size;

        %%to overcome the underflow
        prob_pos = log2(prob_pos);
        prob_neg = log2(prob_neg);

        prob_pos_vec = zeros(10,9);
        prob_neg_vec = zeros(10,9);
        num_features = length(train_X(1,:));   %%number of features of X
        
        %%%training the classifier
        for k  = 1 : num_features 
            for j = 1 : 10
                for i = 1 : train_size
                    if train_y(i) == 1
                        if train_X(i,k) == j
                            prob_pos_vec(j,k) = prob_pos_vec(j,k) + 1;
                        end
                    else
                        if train_X(i,k) == j
                            prob_neg_vec(j,k) = prob_neg_vec(j,k) + 1;
                        end
                    end    
                end        
                prob_pos_vec(j,k) = prob_pos_vec(j,k) / pos_count ;
                prob_neg_vec(j,k) = prob_neg_vec(j,k) / neg_count ;      
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %testing classifier%
        final_prob_pos = 0;
        final_prob_neg = 0;
        for i = 1 : length(test_X)
            final_prob_pos = prob_pos;
            final_prob_neg = prob_neg;
            for j = 1 : num_features
                final_prob_pos = final_prob_pos + log2(prob_pos_vec(test_X(i,j),j)); 
                final_prob_neg = final_prob_neg + log2(prob_neg_vec(test_X(i,j),j));
            end   
            if and(final_prob_pos >= final_prob_neg, test_y(i)==1)
                acc(mainloop) = acc(mainloop) + 1;
            elseif and(final_prob_pos < final_prob_neg, test_y(i)== -1)    
                acc(mainloop) = acc(mainloop) + 1;
            end    
        end
        acc(mainloop) = acc(mainloop) / length(test_X);     %%%accuracy value
        acc(mainloop) = acc(mainloop) + 1;    %%%adding 1 for smoothing
    end
accmean = mean(acc(1:5));   %%%average of accuracies    
disp(accmean);
end



