function [acc,error] = classification_error(pred,target)
    
    acc = sum(pred == target)/numel(target);
    error = 1 - acc; 
    
end