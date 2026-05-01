package knn_dt.evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import knn_dt.algorithm.BaseAlgorithm;
import knn_dt.model.UserRecord;


public class Evaluator {

    
    public EvaluationResult evaluate(BaseAlgorithm classifier,
                                      List<UserRecord> testData) {

        long start = System.currentTimeMillis();

        int correct = 0;
        int total   = testData.size();
        List<String> predictions  = new ArrayList<>();
        List<String> actuals      = new ArrayList<>();
        long totalPredictTime     = 0;

        for (UserRecord record : testData) {
            long t0 = System.currentTimeMillis();
            String predicted = classifier.predict(record);
            totalPredictTime += (System.currentTimeMillis() - t0);

            predictions.add(predicted);
            actuals.add(record.getCategory());

            if (predicted.equalsIgnoreCase(record.getCategory())) correct++;
        }

        long totalTime = System.currentTimeMillis() - start;
        double accuracy = (double) correct / total;

        
        Map<String, Map<String, Integer>> matrix =
                buildConfusionMatrix(actuals, predictions);

        
        Map<String, Double> precision = computePrecision(matrix);
        Map<String, Double> recall    = computeRecall(matrix);
        Map<String, Double> f1        = computeF1(precision, recall);

        return new EvaluationResult(
                classifier.getName(),
                accuracy,
                correct,
                total,
                classifier.getTrainTimeMs(),
                totalPredictTime,
                totalTime,
                matrix,
                precision,
                recall,
                f1
        );
    }

    
    private Map<String, Map<String, Integer>> buildConfusionMatrix(
            List<String> actuals, List<String> predictions) {

        Map<String, Map<String, Integer>> matrix = new LinkedHashMap<>();
        for (int i = 0; i < actuals.size(); i++) {
            String actual = actuals.get(i);
            String pred   = predictions.get(i);
            matrix.computeIfAbsent(actual, k -> new LinkedHashMap<>());
            matrix.get(actual).merge(pred, 1, Integer::sum);
        }
        return matrix;
    }

    
    private Map<String, Double> computePrecision(
            Map<String, Map<String, Integer>> matrix) {

        Map<String, Integer> truePos  = new HashMap<>();
        Map<String, Integer> falsePos = new HashMap<>();

        for (Map.Entry<String, Map<String, Integer>> row : matrix.entrySet()) {
            for (Map.Entry<String, Integer> col : row.getValue().entrySet()) {
                String pred = col.getKey();
                int count   = col.getValue();
                if (pred.equals(row.getKey()))
                    truePos.merge(pred, count, Integer::sum);
                else
                    falsePos.merge(pred, count, Integer::sum);
            }
        }

        Map<String, Double> precision = new LinkedHashMap<>();
        Set<String> allCats = new LinkedHashSet<>(truePos.keySet());
        allCats.addAll(falsePos.keySet());

        for (String cat : allCats) {
            int tp = truePos.getOrDefault(cat, 0);
            int fp = falsePos.getOrDefault(cat, 0);
            precision.put(cat, (tp + fp == 0) ? 0.0 : (double) tp / (tp + fp));
        }
        return precision;
    }

    
    private Map<String, Double> computeRecall(
            Map<String, Map<String, Integer>> matrix) {

        Map<String, Double> recall = new LinkedHashMap<>();
        for (Map.Entry<String, Map<String, Integer>> row : matrix.entrySet()) {
            String actual = row.getKey();
            int tp = row.getValue().getOrDefault(actual, 0);
            int total = row.getValue().values().stream().mapToInt(Integer::intValue).sum();
            recall.put(actual, total == 0 ? 0.0 : (double) tp / total);
        }
        return recall;
    }

    
    private Map<String, Double> computeF1(Map<String, Double> precision,
                                           Map<String, Double> recall) {
        Map<String, Double> f1 = new LinkedHashMap<>();
        Set<String> cats = new LinkedHashSet<>(precision.keySet());
        cats.addAll(recall.keySet());

        for (String cat : cats) {
            double p = precision.getOrDefault(cat, 0.0);
            double r = recall.getOrDefault(cat, 0.0);
            f1.put(cat, (p + r == 0) ? 0.0 : 2 * p * r / (p + r));
        }
        return f1;
    }

    
    public void printReport(EvaluationResult result) {
        System.out.println("\n══════════════════════════════════════════════════");
        System.out.println("  ALGORİTMA: " + result.algorithmName);
        System.out.println("══════════════════════════════════════════════════");
        System.out.printf("  Doğruluk (Accuracy) : %.2f%% (%d / %d)%n",
                result.accuracy * 100, result.correctCount, result.totalCount);
        System.out.printf("  Eğitim Süresi       : %d ms%n", result.trainTimeMs);
        System.out.printf("  Tahmin Süresi (top) : %d ms%n", result.predictTimeMs);
        System.out.printf("  Toplam Süre         : %d ms%n", result.totalTimeMs);
        System.out.println();

        System.out.println("  Sınıf Metrikleri:");
        System.out.printf("  %-20s %-12s %-12s %-12s%n",
                "Kategori", "Precision", "Recall", "F1");
        System.out.println("  " + "-".repeat(56));

        for (String cat : result.precision.keySet()) {
            double p  = result.precision.getOrDefault(cat, 0.0);
            double r  = result.recall.getOrDefault(cat, 0.0);
            double f  = result.f1.getOrDefault(cat, 0.0);
            System.out.printf("  %-20s %-12.4f %-12.4f %-12.4f%n", cat, p, r, f);
        }
        System.out.println("══════════════════════════════════════════════════\n");
    }

    
    public static class EvaluationResult {
        public final String algorithmName;
        public final double accuracy;
        public final int    correctCount;
        public final int    totalCount;
        public final long   trainTimeMs;
        public final long   predictTimeMs;
        public final long   totalTimeMs;
        public final Map<String, Map<String, Integer>> confusionMatrix;
        public final Map<String, Double> precision;
        public final Map<String, Double> recall;
        public final Map<String, Double> f1;

        public EvaluationResult(String name, double acc, int correct, int total,
                                 long trainTime, long predictTime, long totalTime,
                                 Map<String, Map<String, Integer>> matrix,
                                 Map<String, Double> precision,
                                 Map<String, Double> recall,
                                 Map<String, Double> f1) {
            this.algorithmName  = name;
            this.accuracy       = acc;
            this.correctCount   = correct;
            this.totalCount     = total;
            this.trainTimeMs    = trainTime;
            this.predictTimeMs  = predictTime;
            this.totalTimeMs    = totalTime;
            this.confusionMatrix = matrix;
            this.precision      = precision;
            this.recall         = recall;
            this.f1             = f1;
        }
    }
}
