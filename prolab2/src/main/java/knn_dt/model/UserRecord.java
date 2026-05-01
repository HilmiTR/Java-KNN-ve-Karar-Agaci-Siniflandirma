package knn_dt.model;


public class UserRecord {

    private final String clientCode;
    private final String gender;
    private final double lineNetTotal;
    private final String brand;
    private final String brandCode;
    private final String category;

    
    private double genderEncoded;       // Female=0, Male=1
    private double lineNetTotalNorm;    // Min-Max normalize edilmiş
    private double brandCodeEncoded;    // Marka kodu sayısal
    private double[] featureVector;     // Model için hazır özellik vektörü

    public UserRecord(String clientCode, String gender, double lineNetTotal,
                      String brand, String brandCode, String category) {
        this.clientCode = clientCode;
        this.gender = gender;
        this.lineNetTotal = lineNetTotal;
        this.brand = brand;
        this.brandCode = brandCode;
        this.category = category;
    }

    
    public String getClientCode()       { return clientCode; }
    public String getGender()           { return gender; }
    public double getLineNetTotal()     { return lineNetTotal; }
    public String getBrand()            { return brand; }
    public String getBrandCode()        { return brandCode; }
    public String getCategory()         { return category; }

    public double getGenderEncoded()        { return genderEncoded; }
    public double getLineNetTotalNorm()     { return lineNetTotalNorm; }
    public double getBrandCodeEncoded()     { return brandCodeEncoded; }
    public double[] getFeatureVector()      { return featureVector; }

    
    public void setGenderEncoded(double v)      { this.genderEncoded = v; }
    public void setLineNetTotalNorm(double v)   { this.lineNetTotalNorm = v; }
    public void setBrandCodeEncoded(double v)   { this.brandCodeEncoded = v; }
    public void setFeatureVector(double[] v)    { this.featureVector = v; }

    @Override
    public String toString() {
        return String.format("UserRecord{client=%s, gender=%s, lineNetTotal=%.2f, brand=%s, category=%s}",
                clientCode, gender, lineNetTotal, brand, category);
    }
}
