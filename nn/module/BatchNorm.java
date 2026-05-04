package nn.module;

import core.Matrix;
import core.Value;

public class BatchNorm extends ModuleBase {
    private int numFeatures; 

    public BatchNorm(int numFeatures){
        this.numFeatures = numFeatures;
    }

    // precondition: input is a 2D matrix with shape (batch_size, num_features)
    public Matrix forward(Matrix input) {
        int batchSize = input.getRows();
        Matrix output = new Matrix(batchSize, numFeatures);
        for(int i = 0; i < numFeatures; i++){
            double mean = 0.0;
            double variance = 0.0;
            for(int j = 0; j < batchSize; j++){
                mean += input.get(j, i).asNum();
            }
            mean /= batchSize;
            for(int j = 0; j < batchSize; j++){
                double diff = input.get(j, i).asNum() - mean;
                variance += diff * diff;
            }
            variance /= batchSize; 
            for(int j = 0; j < batchSize; j++){
                Value normalized = (input.get(j, i).sub(new Value(mean))
                        .div(new Value(Math.sqrt(1e-5 + variance))));
                output.set(j, i, normalized);
            }
        }
        return output;
    }
}
