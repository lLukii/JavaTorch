package nn.module;

import core.Matrix;
import nn.Functional;

public class ReLU extends ModuleBase {
    public Matrix forward(Matrix input){
        return Functional.relu(input);
    }
}
