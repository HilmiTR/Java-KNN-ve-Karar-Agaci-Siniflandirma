package knn_dt.algorithm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import knn_dt.model.UserRecord;


public abstract class BaseAlgorithm implements IClassifier {

    protected List<UserRecord> trainingData;
    protected long trainTimeMs;
    protected long lastPredictTimeMs;

    

    
    public double calculateAccuracy(List<UserRecord> testData) {
        if (testData == null || testData.isEmpty()) return 0.0;

        int correct = 0;
        for (UserRecord record : testData) {
            String predicted = predict(record);
            if (predicted.equalsIgnoreCase(record.getCategory())) {
                correct++;
            }
        }
        return (double) correct / testData.size();
    }

    

    
    public static List<List<UserRecord>> splitData(List<UserRecord> allData,
                                                    double splitRatio, long seed) {
        List<UserRecord> shuffled = new ArrayList<>(allData);
        Collections.shuffle(shuffled, new Random(seed));

        int trainSize = (int) (shuffled.size() * splitRatio);
        List<UserRecord> train = new ArrayList<>(shuffled.subList(0, trainSize));
        List<UserRecord> test  = new ArrayList<>(shuffled.subList(trainSize, shuffled.size()));

        List<List<UserRecord>> result = new ArrayList<>();
        result.add(train);
        result.add(test);
        return result;
    }

    

    
    public Map<String, Map<String, Integer>> confusionMatrix(List<UserRecord> testData) {
        Map<String, Map<String, Integer>> matrix = new LinkedHashMap<>();

        for (UserRecord record : testData) {
            String actual    = record.getCategory();
            String predicted = predict(record);

            matrix.computeIfAbsent(actual, k -> new LinkedHashMap<>());
            matrix.get(actual).merge(predicted, 1, Integer::sum);
        }
        return matrix;
    }

    
    public long getTrainTimeMs()        { return trainTimeMs; }
    public long getLastPredictTimeMs()  { return lastPredictTimeMs; }
}
