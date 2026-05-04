package examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import backward.Adam;
import core.Matrix;
import core.Value;
import data.DataLoader;
import data.DataSet;
import data.ModelParser;
import nn.Functional;
import nn.module.*;

public class Main {
    public static class NeuralNetwork extends ModuleBase{
        private Sequential model; 
        public NeuralNetwork(){
            model = new Sequential(
                new Linear(784, 64), 
                new ReLU(),
                new Linear(64, 10)
            );
            addRequirements();
        }

        public Matrix forward(Matrix input){
            return Functional.softmax(model.forward(input));
        }
    }

    public static final int NUM_EPOCHS = 15;
    public static final double LR = 0.001;
    public static final int TRAIN_SIZE = 1000, TEST_SIZE = 1000;

    public static void loadData(Matrix input, Matrix target, String path){
        try{
            BufferedReader br = new BufferedReader(
                new FileReader(new File(path)));
            String line;
            int row = 0;
            while((line = br.readLine()) != null){
                if(row > 0){ // skip header
                    String[] values = line.split(",");
                    for(int i = 0; i < values.length - 1; i++){
                        double value = Double.parseDouble(values[i]);
                        input.set(row - 1, i, new Value(value / 255.0));
                    }
                    int label = Integer.parseInt(values[values.length - 1]);
                    target.set(row - 1, 0, new Value(label));
                }
                row++;
            }
            br.close();
        }
        catch(IOException e){
            System.out.println("Error reading input file: " + e);
        }
    }

    public static class FashionDataset extends DataSet{
        private Matrix inputs; 
        private Matrix labels; 

        FashionDataset(Matrix inputs, Matrix labels){
            this.inputs = inputs;
            this.labels = labels;
        }

        public int getSize(){
            return labels.getRows();
        }

        public Matrix[] getItem(int index){
            return new Matrix[]{
                inputs.get(index, index, 0, inputs.getCols() - 1), 
                labels.get(index, index, 0, 0)
            };
        }
    }

    public static void trainLoop(NeuralNetwork model, DataLoader trainDl, CrossEntropyLoss lossFn, Adam optim){
        Matrix[] batch;
        double totalLoss = 0.0;
        int numBatches = 0;
        trainDl.resetIterator();
        while((batch = trainDl.getNext()) != null){
            Matrix inputs = batch[0];
            Matrix labels = batch[1];
            Matrix outputs = model.forward(inputs);
            Value loss = lossFn.forward(outputs, labels);
            totalLoss += loss.asNum();
            loss.backward();
            optim.step();
            numBatches++;
        }
        System.out.println("Loss: " + totalLoss / numBatches);
    }

    public static void evalLoop(NeuralNetwork model, DataLoader testDl){
        int correct = 0;
        int total = 0;
        Matrix[] batch;
        testDl.resetIterator();
        while((batch = testDl.getNext()) != null){
            Matrix inputs = batch[0];
            Matrix labels = batch[1];
            Matrix outputs = model.forward(inputs);
            for(int i = 0; i < outputs.getRows(); i++){
                int pred = 0;
                double maxProb = outputs.get(i, 0).asNum();
                for(int j = 1; j < outputs.getCols(); j++){
                    if(outputs.get(i, j).asNum() > maxProb){
                        maxProb = outputs.get(i, j).asNum();
                        pred = j;
                    }
                }
                if(pred == (int)labels.get(i, 0).asNum()){
                    correct++;
                }
                total++;
            }
        }
        System.out.println("Test Accuracy: " + (100.0 * correct) / total + "%");
    }

    public static void main(String[] args) throws IOException{
        NeuralNetwork model = new NeuralNetwork();
        Matrix trainData = new Matrix(1000, 784); 
        Matrix trainLabels = new Matrix(1000, 1);
        Matrix testData = new Matrix(1000, 784); 
        Matrix testLabels = new Matrix(1000, 1);
        CrossEntropyLoss lossFn = new CrossEntropyLoss();
        Adam optim = new Adam(model.parameters(), LR);

        loadData(trainData, trainLabels, "examples/datasets/fashion_train.csv");
        loadData(testData, testLabels, "examples/datasets/fashion_test.csv");
        FashionDataset trainDataset = new FashionDataset(trainData, trainLabels);
        FashionDataset testDataset = new FashionDataset(testData, testLabels);
        DataLoader trainDl = new DataLoader(trainDataset, 16);
        DataLoader testDl = new DataLoader(testDataset, 16);

        for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){
            trainLoop(model, trainDl, lossFn, optim);
            evalLoop(model, testDl);
        }

        ModelParser.saveModel(model, "examples/saved/test");
    }
}
