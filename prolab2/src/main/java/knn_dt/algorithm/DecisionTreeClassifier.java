package knn_dt.algorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import knn_dt.model.UserRecord;


public class DecisionTreeClassifier extends BaseAlgorithm {

    private int maxDepth;
    private int minSamples;
    private TreeNode root;

    
    private static final String[] FEATURE_NAMES =
            { "genderEncoded", "lineNetTotalNorm", "brandCodeEncoded" };

    public DecisionTreeClassifier(int maxDepth, int minSamples) {
        this.maxDepth   = maxDepth;
        this.minSamples = minSamples;
    }

    @Override
    public String getName() { return "Decision Tree (depth=" + maxDepth + ")"; }

    
    @Override
    public void train(List<UserRecord> trainingData) {
        long start = System.currentTimeMillis();
        this.trainingData = new ArrayList<>(trainingData);
        this.root = buildTree(trainingData, 0);
        this.trainTimeMs = System.currentTimeMillis() - start;
    }

    
    @Override
    public String predict(UserRecord record) {
        long start = System.currentTimeMillis();
        String result = traverseTree(root, record);
        lastPredictTimeMs = System.currentTimeMillis() - start;
        return result;
    }

    
    private TreeNode buildTree(List<UserRecord> data, int depth) {
        
        if (data.isEmpty()) return new TreeNode(mostCommonCategory(trainingData));
        if (depth >= maxDepth || data.size() < minSamples || isPure(data)) {
            return new TreeNode(mostCommonCategory(data));
        }

        
        BestSplit best = findBestSplit(data);
        if (best == null) return new TreeNode(mostCommonCategory(data));

       
        List<UserRecord> leftData  = new ArrayList<>();
        List<UserRecord> rightData = new ArrayList<>();
        for (UserRecord r : data) {
            if (r.getFeatureVector()[best.featureIdx] <= best.threshold)
                leftData.add(r);
            else
                rightData.add(r);
        }

        if (leftData.isEmpty() || rightData.isEmpty())
            return new TreeNode(mostCommonCategory(data));

        
        TreeNode node = new TreeNode(best.featureIdx, best.threshold,
                FEATURE_NAMES[best.featureIdx]);
        node.left  = buildTree(leftData,  depth + 1);
        node.right = buildTree(rightData, depth + 1);
        return node;
    }

    
    private String traverseTree(TreeNode node, UserRecord record) {
        if (node.isLeaf()) return node.label;

        double featureVal = record.getFeatureVector()[node.featureIdx];
        if (featureVal <= node.threshold)
            return traverseTree(node.left, record);
        else
            return traverseTree(node.right, record);
    }

    
    private BestSplit findBestSplit(List<UserRecord> data) {
        double baseEntropy = entropy(data);
        double bestGain    = -1;
        BestSplit best     = null;

        int numFeatures = data.get(0).getFeatureVector().length;

        for (int fi = 0; fi < numFeatures; fi++) {
            
            Set<Double> thresholds = new TreeSet<>();
            for (UserRecord r : data)
                thresholds.add(r.getFeatureVector()[fi]);

            for (double threshold : thresholds) {
                List<UserRecord> left  = new ArrayList<>();
                List<UserRecord> right = new ArrayList<>();
                for (UserRecord r : data) {
                    if (r.getFeatureVector()[fi] <= threshold) left.add(r);
                    else right.add(r);
                }
                if (left.isEmpty() || right.isEmpty()) continue;

                double weightedEntropy =
                        ((double) left.size()  / data.size()) * entropy(left) +
                        ((double) right.size() / data.size()) * entropy(right);

                double gain = baseEntropy - weightedEntropy;
                if (gain > bestGain) {
                    bestGain = gain;
                    best = new BestSplit(fi, threshold);
                }
            }
        }
        return best;
    }

    
    private double entropy(List<UserRecord> data) {
        if (data.isEmpty()) return 0.0;

        Map<String, Integer> counts = new HashMap<>();
        for (UserRecord r : data)
            counts.merge(r.getCategory(), 1, Integer::sum);

        double ent = 0.0;
        int n = data.size();
        for (int count : counts.values()) {
            double p = (double) count / n;
            if (p > 0) ent -= p * (Math.log(p) / Math.log(2));
        }
        return ent;
    }

    
    private boolean isPure(List<UserRecord> data) {
        String first = data.get(0).getCategory();
        for (UserRecord r : data)
            if (!r.getCategory().equals(first)) return false;
        return true;
    }

    private String mostCommonCategory(List<UserRecord> data) {
        Map<String, Integer> counts = new HashMap<>();
        for (UserRecord r : data)
            counts.merge(r.getCategory(), 1, Integer::sum);
        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("Unknown");
    }

   

    
    private static class TreeNode {
        String label;       
        int    featureIdx;  
        double threshold;   
        String featureName; 
        TreeNode left, right;

        
        TreeNode(String label) {
            this.label = label;
        }

        
        TreeNode(int featureIdx, double threshold, String featureName) {
            this.featureIdx  = featureIdx;
            this.threshold   = threshold;
            this.featureName = featureName;
        }

        boolean isLeaf() { return label != null; }

        @Override
        public String toString() {
            return isLeaf()
                    ? "Leaf[" + label + "]"
                    : "Node[" + featureName + " <= " + String.format("%.4f", threshold) + "]";
        }
    }

    
    private static class BestSplit {
        int    featureIdx;
        double threshold;

        BestSplit(int featureIdx, double threshold) {
            this.featureIdx = featureIdx;
            this.threshold  = threshold;
        }
    }

    
    public int getMaxDepth()   { return maxDepth; }
    public int getMinSamples() { return minSamples; }
}
