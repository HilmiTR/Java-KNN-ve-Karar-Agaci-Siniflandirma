package knn_dt.algorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import knn_dt.model.UserRecord;


public class KNNClassifier extends BaseAlgorithm {

    private int k;

    public KNNClassifier(int k) {
        this.k = k;
    }

    @Override
    public String getName() {
        return "KNN (k=" + k + ")";
    }

    
    @Override
    public void train(List<UserRecord> trainingData) {
        long start = System.currentTimeMillis();
        this.trainingData = new ArrayList<>(trainingData);
        this.trainTimeMs  = System.currentTimeMillis() - start;
    }

    
    @Override
    public String predict(UserRecord query) {
        long start = System.currentTimeMillis();

        
        PriorityQueue<double[]> pq = new PriorityQueue<>(
                Comparator.comparingDouble(a -> a[0]));

        for (int i = 0; i < trainingData.size(); i++) {
            double dist = euclideanDistance(query.getFeatureVector(),
                                            trainingData.get(i).getFeatureVector());
            pq.offer(new double[]{ dist, i });
        }

        
        Map<String, Integer> votes = new HashMap<>();
        for (int i = 0; i < k && !pq.isEmpty(); i++) {
            int idx = (int) pq.poll()[1];
            String cat = trainingData.get(idx).getCategory();
            votes.merge(cat, 1, Integer::sum);
        }

        
        lastPredictTimeMs = System.currentTimeMillis() - start;
        return votes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("Unknown");
    }

    
    private double euclideanDistance(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length)
            return Double.MAX_VALUE;

        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    
    public int getK()       { return k; }
    public void setK(int k) { this.k = k; }
}
