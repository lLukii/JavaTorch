package nn.module;

import core.Matrix;
import core.Value;

public abstract class LossBase {
    public LossBase() {}
    
    public abstract Value forward(Matrix pred, Matrix target); 
}
