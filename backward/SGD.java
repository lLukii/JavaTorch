package backward;

import core.Value;
import nn.module.ModuleBase;

import java.util.ArrayList;
import java.util.List;

/**
 * Stochastic Gradient Descent optimizer that updates parameters based on their gradients and a specified learning rate.
 * @author lLukii
 */
public class SGD {
    private double lr, momentum; 
    private int step = 1;
    private List<Value> parameters;
    private List<Double> velocity = new ArrayList<>();

    public SGD(List<Value> parameters, double lr, double momentum){
        this.lr = lr;
        this.momentum = momentum;
        this.parameters = parameters;
        for(int i = 0; i < parameters.size(); i++){
            velocity.add(0.0);
        }
    }

    public SGD(List<Value> parameters, double lr){
        this(parameters, lr, 0.0);
    }

    public void step(){
       for(int i = 0; i < parameters.size(); i++){
           Value param = parameters.get(i);
           double grad = param.getGrad();
           double v = velocity.get(i);
           v = momentum * v - lr * grad;
           velocity.set(i, v);
           param.changeData(v);
       }
       step++;
    }
}
