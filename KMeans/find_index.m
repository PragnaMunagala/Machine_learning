%This function finds the row number of the data whose cluster = idx.
function ind = find_index(cluster_matrix,idx)
    ind = [];
    for i = 1 : 210
        if cluster_matrix(i,8) == idx
            ind = [ind i];
        end    
    end    
end