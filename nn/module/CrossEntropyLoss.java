package nn.module;

import core.Matrix;
import core.Value;

public class CrossEntropyLoss extends LossBase {

    public Value forward(Matrix pred, Matrix target){
        int size = pred.getRows();
        int nDim = pred.getCols();
        Value loss = new Value(0.0);

        for(int i = 0; i < size; i++){
            int targetClass = (int) target.get(i, 0).asNum();
            Value logProb = pred.get(i, targetClass).log();
            loss = loss.sub(logProb);
        }

        return loss.div(new Value(size));
    }
}
