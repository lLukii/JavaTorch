package nn.module;

import core.Matrix;

/**
 * Linear layer that performs a linear transformation on the input data using weights. The forward method computes the output by multiplying the input and weight matrices. (X' = XW) 
 * 
 * @author lLukii
 */
public class Linear extends ModuleBase {
    private Matrix weights; 

    public Linear(int inputSize, int outputSize){
        weights = new Matrix(inputSize, outputSize);
        weights.fill();
        addRequirements();
    }

    public Matrix forward(Matrix input){
        return input.matMul(weights);
    }
}