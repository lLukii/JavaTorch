package nn;

import core.Matrix;
import core.Value;

public final class Functional {
    public static Matrix relu(Matrix m){
        for(int i = 0; i < m.getRows(); i++){
            for(int j = 0; j < m.getCols(); j++){
                m.set(i, j, m.get(i, j).relu());
            }
        }
        return m;
    }

    public static Matrix softmax(Matrix logits){
        // dim: [batch, output]
        int numBatches = logits.getRows();
        int outputDim = logits.getCols();
        Matrix output = new Matrix(numBatches, outputDim);
        for(int i = 0; i < numBatches; i++){
            Value[] expVals = new Value[outputDim];
            Value sum = new Value(0);

            for(int j = 0; j < outputDim; j++){
                expVals[j] = logits.get(i, j).exp();
                sum = sum.add(expVals[j]);
            }
            
            for(int j = 0; j < outputDim; j++){
                output.set(i, j, expVals[j].div(sum));
            }
        }
        return output;
    }
}
