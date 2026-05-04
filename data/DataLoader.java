package data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import core.Matrix;

public class DataLoader {
    private List<Matrix[]> batches;
    private Iterator<Matrix[]> batchIterator;

    public DataLoader(DataSet dataSet, int batchSize){
        batches = new ArrayList<>();
        int inputDim = dataSet.getItem(0)[0].getCols(); 
        // input dimensionality should be the same across all data points
        for(int i = 0; i < dataSet.getSize(); i += batchSize){
            int curBatchSize = Math.min(batchSize, dataSet.getSize() - i);
            Matrix batchInput = new Matrix(curBatchSize, inputDim);
            Matrix batchLabels = new Matrix(curBatchSize, 1);

            for(int j = 0; j < curBatchSize; j++){
                Matrix[] dataPair = dataSet.getItem(i + j);
                for(int k = 0; k < inputDim; k++){
                    batchInput.set(j, k, dataPair[0].get(0, k));
                }
                batchLabels.set(j, 0, dataPair[1].get(0, 0));
            }

            batches.add(new Matrix[]{batchInput, batchLabels});
            resetIterator();
        }
    }

    public Matrix[] getNext(){
        return batchIterator.hasNext() ? batchIterator.next() : null;
    }

    public void resetIterator(){
        batchIterator = batches.iterator();
    }
}
