%parsing the text file and forming the feature matrix
s = dlmread('seeds_dataset.txt');
train_X = s(:,1:8);
[rows,columns] = size(train_X); 
flag=0;
final_obj= [];
temp_matrix = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Training start
for k = 2 : 10 
    means = [];   
    %Assign random points as initial centroid vector
    for i = 1 : k
        means = [means ; train_X(randi([1 210]),:)];
    end 
    %Loop till kmeans algorithm converges
    while 1        
        %Assign the label to the point to the closest centroid vector.
        cluster_matrix = [];
        for i = 1 : rows
            aprev = realmax;
            acurr = 0;
            for j = 1 : k
                acurr = sqrt(sum((means(j,1:7) - train_X(i,1:7)) .^ 2));
                if(acurr < aprev)
                    aprev = acurr;
                    flag = j;
                end           
            end           
            cluster_matrix = [cluster_matrix;train_X(i,1:7) flag];
        end        
        means = [];      
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Break a big cluster into small clusters if a cluster has no
        %datapoints.
        temp = [];
        vec = [];
        for i = 1 : k
            ind = cluster_matrix(:,8) == i;
            [r,c] = size(cluster_matrix(ind,:));
            vec = [vec r];
        end
        [val idx] = max(vec);      
        main_index = find(vec==0);  
        if ~isempty(main_index)            
            temp_index = find_index(cluster_matrix,idx);
            w = int16(length(temp_index) / (length(main_index)+1));
            counter = 1;
            for q = 1 : length(temp_index) - w
                if q < counter * w
                    cluster_matrix(temp_index(q),8) = main_index(counter);
                else
                    counter = counter + 1;                  
                end
            end            
        end    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
        %Recalculate the centroid vector and check for convergence
        %If not converged, then repeat. 
        %Else find the objective function.
        temp = [];
        for i = 1 : k
            ind = cluster_matrix(:,8) == i;
            temp = mean(cluster_matrix(ind,:));
            means = [means ; temp];
        end
        if isequal(cluster_matrix,temp_matrix)
            break;
        else
            temp_matrix = cluster_matrix;
        end    
    end 
    obj = 0;
    %Find the objective function
    for p = 1 : rows
        obj = obj + sum((means(cluster_matrix(p,8),1:7) - cluster_matrix(p,1:7)).^2);
    end    
    final_obj = [final_obj obj];    
end
plot([2:10],final_obj);
xlim([1 10]);
xlabel('Number of clusters');
xlabel('Value of objective function');
title('#clusters vs Objective function');