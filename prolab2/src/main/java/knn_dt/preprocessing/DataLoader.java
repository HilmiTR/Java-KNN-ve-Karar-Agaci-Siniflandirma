package knn_dt.preprocessing;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import knn_dt.model.UserRecord;


public class DataLoader {

    
    private static final int COL_CLIENT_CODE = 9;
private static final int COL_GENDER      = 17;
private static final int COL_LINE_NET    = 7;
private static final int COL_BRAND       = 11;
private static final int COL_BRAND_CODE  = 10;
private static final int COL_CATEGORY    = 12;

    
    public List<UserRecord> loadFromXlsx(String filePath) throws IOException {
        List<UserRecord> records = new ArrayList<>();
        int skipped = 0;

        try (FileInputStream fis = new FileInputStream(filePath);
             Workbook workbook   = new XSSFWorkbook(fis)) {

            Sheet sheet = workbook.getSheetAt(0);

            
            boolean firstRow = true;
            for (Row row : sheet) {
                if (firstRow) { firstRow = false; continue; }

                try {
                    UserRecord record = parseRow(row);
                    if (record != null) records.add(record);
                    else skipped++;
                } catch (Exception e) {
                    System.out.println("Okuma Hatası: " + e.getMessage());
                    
                    skipped++;
                }
            }
        }

        System.out.printf("[DataLoader] %d kayıt yüklendi, %d satır atlandı.%n",
                records.size(), skipped);
        return records;
    }

    
    public List<UserRecord> loadFromCsv(String filePath) throws IOException {
        List<UserRecord> records = new ArrayList<>();
        int skipped = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                if (firstLine) { firstLine = false; continue; }

                try {
                    String[] parts = line.split(",", -1);
                    if (parts.length < 6) { skipped++; continue; }

                    String clientCode  = parts[COL_CLIENT_CODE].trim();
                    String gender      = parts[COL_GENDER].trim();
                    double lineNetTotal = Double.parseDouble(parts[COL_LINE_NET].trim());
                    String brand       = parts[COL_BRAND].trim();
                    String brandCode   = parts[COL_BRAND_CODE].trim();
                    String category    = parts[COL_CATEGORY].trim();

                    if (clientCode.isEmpty() || gender.isEmpty() || category.isEmpty()) {
                        skipped++; continue;
                    }

                    records.add(new UserRecord(clientCode, gender, lineNetTotal,
                                               brand, brandCode, category));
                } catch (NumberFormatException e) {
                    skipped++;
                }
            }
        }

        System.out.printf("[DataLoader] %d kayıt yüklendi, %d satır atlandı.%n",
                records.size(), skipped);
        return records;
    }

    
    private UserRecord parseRow(Row row) {
        try {
            String clientCode   = getCellString(row, COL_CLIENT_CODE);
            String gender       = getCellString(row, COL_GENDER);
            double lineNetTotal = getCellDouble(row, COL_LINE_NET);
            String brand        = getCellString(row, COL_BRAND);
            String brandCode    = getCellString(row, COL_BRAND_CODE);
            String category     = getCellString(row, COL_CATEGORY);

            
            if (clientCode == null || clientCode.isEmpty()) return null;
            if (gender == null     || gender.isEmpty())     return null;
            if (category == null   || category.isEmpty())   return null;
            if (brand == null || brand.isEmpty()) return null;
            if (brandCode == null || brandCode.isEmpty()) return null;

            return new UserRecord(clientCode, gender, lineNetTotal,
                                  brand, brandCode, category);

        } catch (Exception e) {
            System.out.println("Asıl Hata: " + e.toString());
            return null;
        }
    }

    private String getCellString(Row row, int col) {
        Cell cell = row.getCell(col);
        if (cell == null) return null;

        switch (cell.getCellType()) {
            case STRING:  return cell.getStringCellValue().trim();
            case NUMERIC: return String.valueOf((long) cell.getNumericCellValue());
            default:      return null;
        }
    }

    private double getCellDouble(Row row, int col) {
        Cell cell = row.getCell(col);
        if (cell == null) throw new IllegalArgumentException("Null cell at col " + col);

        switch (cell.getCellType()) {
            case NUMERIC: return cell.getNumericCellValue();
            case STRING:  return Double.parseDouble(cell.getStringCellValue().trim());
            default: throw new IllegalArgumentException("Non-numeric cell at col " + col);
        }
    }
}
