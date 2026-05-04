package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

import core.Parameter;
import core.Value;
import nn.module.ModuleBase;

public class ModelParser {
    public static void saveModel(ModuleBase model, String path) throws IOException {
        for(Map.Entry<String, Parameter> entry : model.getStateDict().entrySet()){
            String filePath = path + "/" + entry.getKey() + "_params.txt";
            File file = new File(filePath);
            if(!file.exists() && file.createNewFile()){
                BufferedWriter writer = new BufferedWriter(new FileWriter(file));
                Parameter param = entry.getValue();
                for(int i = 0; i < param.getRows(); i++){
                    for(int j = 0; j < param.getCols(); j++){
                        writer.write(param.get(i, j).asNum() + ",");
                    }
                    writer.write("\n");
                }
                writer.close();
            }
        }
        for(Map.Entry<String, ModuleBase> entry : model.getSubModules().entrySet()){
            String subModulePath = path + "/" + entry.getKey();
            File subModuleDir = new File(subModulePath);
            if(!subModuleDir.exists()){
                subModuleDir.mkdir();
            }
            saveModel(entry.getValue(), subModulePath);
        }
    }

    public static void loadModel(ModuleBase model, String path) throws IOException {
        for(Map.Entry<String, Parameter> entry : model.getStateDict().entrySet()){
            String filePath = path + "/" + entry.getKey() + "_params.txt";
            File file = new File(filePath);
            if(file.exists()){
                BufferedReader reader = new BufferedReader(new FileReader(file));
                Parameter param = entry.getValue();
                String line;
                int row = 0;
                while((line = reader.readLine()) != null){
                    String[] values = line.split(",");
                    for(int col = 0; col < values.length; col++){
                        double value = Double.parseDouble(values[col]);
                        param.set(row, col, new Value(value));
                    }
                    row++;
                }
                reader.close();
            }
        }
        for(Map.Entry<String, ModuleBase> entry : model.getSubModules().entrySet()){
            String subModulePath = path + "/" + entry.getKey();
            loadModel(entry.getValue(), subModulePath);
        }
    }
}
