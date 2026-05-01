package knn_dt.algorithm;

import java.util.List;

import knn_dt.model.UserRecord;


public interface IClassifier {

    
    void train(List<UserRecord> trainingData);

    
    String predict(UserRecord record);

    
    String getName();
}
