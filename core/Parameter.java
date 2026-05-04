package core;

import java.util.Random;

/**
 * Matrix wrapper that prevents standard matrices defined in a model as trainable parameters, offering
 * more versatility to the user. 
 * Initializes the matrix with random gaussian values. 
 * 
 * @author lLukii
 */
public class Parameter extends Matrix{
    public Parameter(int rows, int cols){
        super(rows, cols);
        Random rand = new Random();
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                this.set(i, j, new Value(rand.nextGaussian(0, 0.05))); // Random values between -1 and 1
            }
        }
    }   
}
