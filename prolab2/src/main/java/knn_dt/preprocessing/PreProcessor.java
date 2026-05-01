package knn_dt.preprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import knn_dt.model.UserRecord;


public class PreProcessor {

    
    private double minLineNet, maxLineNet;
    private double minBrandCode, maxBrandCode;

    
    private final Map<String, Double> brandCodeMap = new HashMap<>();

    
    public void fitAndTransform(List<UserRecord> trainingData, List<UserRecord> allData) {
        
        computeStats(trainingData);

        
        for (UserRecord record : allData) {
            encodeRecord(record);
        }
    }

    
    private void computeStats(List<UserRecord> data) {
        minLineNet   = Double.MAX_VALUE;
        maxLineNet   = Double.MIN_VALUE;
        minBrandCode = Double.MAX_VALUE;
        maxBrandCode = Double.MIN_VALUE;

        
        Set<String> brandCodes = new LinkedHashSet<>();
        for (UserRecord r : data) {
            double lnt = r.getLineNetTotal();
            if (lnt < minLineNet) minLineNet = lnt;
            if (lnt > maxLineNet) maxLineNet = lnt;
            brandCodes.add(r.getBrandCode());
        }

        
        List<String> sortedCodes = new ArrayList<>(brandCodes);
        Collections.sort(sortedCodes);
        for (int i = 0; i < sortedCodes.size(); i++) {
            brandCodeMap.put(sortedCodes.get(i), (double) i);
        }

        minBrandCode = 0;
        maxBrandCode = Math.max(1, sortedCodes.size() - 1);
    }

    
    private void encodeRecord(UserRecord r) {
        
        double genderEnc = r.getGender().equalsIgnoreCase("male") ? 1.0 : 0.0;
        r.setGenderEncoded(genderEnc);

        
        double lntNorm = normalize(r.getLineNetTotal(), minLineNet, maxLineNet);
        r.setLineNetTotalNorm(lntNorm);

        
        double bcNum = brandCodeMap.getOrDefault(r.getBrandCode(), 0.0);
        r.setBrandCodeEncoded(bcNum);
        double bcNorm = normalize(bcNum, minBrandCode, maxBrandCode);

        
        r.setFeatureVector(new double[]{ genderEnc, lntNorm, bcNorm });
    }

    
    private double normalize(double value, double min, double max) {
        if (Math.abs(max - min) < 1e-9) return 0.0;
        return (value - min) / (max - min);
    }

    
    public double getMinLineNet()   { return minLineNet; }
    public double getMaxLineNet()   { return maxLineNet; }
    public int    getBrandCount()   { return brandCodeMap.size(); }
}
