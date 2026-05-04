package backward;

import java.util.List;

import core.Value;

/**
 * Adam optimizer that updates parameters based on their gradients, first and second moment estimates, and specified hyperparameters. Based on the paper "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba.
 * @author lLukii
 */
public class Adam {
    private double lr;
    private double beta1;
    private double beta2;
    private double epsilon;

    private double[] m; 
    private double[] v;
    private int step = 1; 
    private List<Value> parameters;

    public Adam(List<Value> parameters, double lr, double beta1, double beta2, double epsilon){
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.parameters = parameters;

        m = new double[parameters.size()];
        v = new double[parameters.size()];
        for(int i = 0; i < m.length; i++){
            m[i] = 0;
            v[i] = 0;
        }
    }

    public Adam(List<Value> parameters, double lr){
        this(parameters, lr, 0.9, 0.999, 1e-8);
    }

    public void step(){
        for(int i = 0; i < parameters.size(); i++){
            m[i] = beta1 * m[i] + (1 - beta1) * parameters.get(i).getGrad();
            v[i] = beta2 * v[i] + (1 - beta2) * Math.pow(parameters.get(i).getGrad(), 2);
            double mHat = m[i] / (1 - Math.pow(beta1, step));
            double vHat = v[i] / (1 - Math.pow(beta2, step));
            double update = lr * mHat / (Math.sqrt(vHat) + epsilon);
            Value p = parameters.get(i);
            p.changeData(-update);
            p.setGrad(0);
        }
        step++;
    }
}
