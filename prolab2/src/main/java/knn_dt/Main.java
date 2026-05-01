package knn_dt;

import java.util.ArrayList;
import java.util.List;

import knn_dt.algorithm.BaseAlgorithm;
import knn_dt.algorithm.DecisionTreeClassifier;
import knn_dt.algorithm.IClassifier;
import knn_dt.algorithm.KNNClassifier;
import knn_dt.evaluation.Evaluator;
import knn_dt.model.UserRecord;
import knn_dt.preprocessing.DataLoader;
import knn_dt.preprocessing.PreProcessor;


public class Main {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║   ProLab-II: KNN vs Karar Ağacı Karşılaştırması ║");
        System.out.println("╚══════════════════════════════════════════════════╝\n");

        
        String filePath  = args.length > 0 ? args[0] : "data/MarketSalesKocaeli.xlsx";
        int    kValue    = args.length > 1 ? Integer.parseInt(args[1]) : 5;
        int    maxDepth  = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        double splitRatio= args.length > 3 ? Double.parseDouble(args[3]) : 0.8;

        System.out.printf("Parametreler: K=%d, maxDepth=%d, splitRatio=%.0f%%%n%n",
                kValue, maxDepth, splitRatio * 100);

        try {
            
            System.out.println("[1/5] Veri yükleniyor: " + filePath);
            DataLoader loader = new DataLoader();
            List<UserRecord> allData;

            if (filePath.toLowerCase().endsWith(".csv")) {
                allData = loader.loadFromCsv(filePath);
            } else {
                allData = loader.loadFromXlsx(filePath);
            }

            if (allData.isEmpty()) {
                System.err.println("Hata: Veri seti boş!");
                return;
            }

           
            System.out.printf("%n[2/5] Veri bölünüyor (%.0f%% eğitim / %.0f%% test)...%n",
                    splitRatio * 100, (1 - splitRatio) * 100);

            List<List<UserRecord>> splits = BaseAlgorithm.splitData(allData, splitRatio, 42L);
            List<UserRecord> trainData = splits.get(0);
            List<UserRecord> testData  = splits.get(1);

            System.out.printf("Eğitim: %d kayıt, Test: %d kayıt%n",
                    trainData.size(), testData.size());

            
            System.out.println("\n[3/5] Veri ön işleme (encoding + normalizasyon)...");
            PreProcessor preProcessor = new PreProcessor();
            preProcessor.fitAndTransform(trainData, allData);
            System.out.printf("Marka sayısı: %d, LineNetTotal aralığı: [%.2f, %.2f]%n",
                    preProcessor.getBrandCount(),
                    preProcessor.getMinLineNet(),
                    preProcessor.getMaxLineNet());

            
            System.out.println("\n[4/5] Algoritmalar eğitiliyor...");

            
            List<IClassifier> classifiers = new ArrayList<>();
            classifiers.add(new KNNClassifier(kValue));
            classifiers.add(new DecisionTreeClassifier(maxDepth, 5));

            for (IClassifier clf : classifiers) {
                System.out.printf("  Eğitiliyor: %s ... ", clf.getName());
                ((BaseAlgorithm) clf).train(trainData);
                System.out.printf("tamamlandı (%d ms)%n",
                        ((BaseAlgorithm) clf).getTrainTimeMs());
            }

            
            System.out.println("\n[5/5] Değerlendirme yapılıyor...\n");
            Evaluator evaluator = new Evaluator();

            for (IClassifier clf : classifiers) {
                Evaluator.EvaluationResult result =
                        evaluator.evaluate((BaseAlgorithm) clf, testData);
                evaluator.printReport(result);
            }

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
