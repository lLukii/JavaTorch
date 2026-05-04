package core;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


/**
 * Gradient-influenced values that hold a double and its corresponding gradient. <br>
 * Largely based on Andrej Karpathy's micrograd implementation. 
 * 
 * @author lLukii
 */
public class Value {
    private double data, grad;
    private double[] local_grads; 
    private Value[] children;

    /**
     * Constructor for Value.
     * @param data - numeric value to be stored
     * @param children - children of this value in the computational graph
     * @param local_grads - derivate wrt to each child
     */
    public Value(double data, Value[] children, double[] local_grads){
        this.data = data;
        this.grad = 0.0;
        this.children = children;
        this.local_grads = local_grads;
    }

    public Value(double data){
        this(data, new Value[0], new double[0]);
    }

    public Value add(Value other){
        return new Value(this.data + other.data, new Value[]{this, other},
            new double[]{1.0, 1.0}
        );
    }

    public Value sub(Value other){
        return new Value(this.data - other.data, new Value[]{this, other},
            new double[]{1.0, -1.0}
        );
    }

    public Value mul(Value other){
        return new Value(this.data * other.data, new Value[]{this, other},
            new double[]{other.data, this.data}
        );
    }

    public Value pow(Value other){
        return new Value(Math.pow(this.data, other.data), new Value[]{this},
            new double[]{other.data * Math.pow(this.data, other.data - 1)}
        );
    }

    public Value log(){
        return new Value(Math.log(this.data), new Value[]{this},
            new double[]{1 / this.data}
        );
    }

    public Value exp(){
        return new Value(Math.exp(this.data), new Value[]{this},
            new double[]{Math.exp(this.data)}
        );
    }

    public Value relu(){
        return new Value(Math.max(0, this.data), new Value[]{this},
            new double[]{this.data > 0 ? 1.0 : 0.0}
        );
    }

    public Value neg(){
        return mul(new Value(-1.0));
    }

    public Value div(Value other){
        return this.mul(other.pow(new Value(-1)));
    }

    /**
     * Performs backpropagation to compute gradients for all values in the computational graph. <br>
     * Precondition: the graph must be a DAG.
     * @param lr - learning rate for gradient descent 
     */
    public void backward(){
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        topoSort(this, topo, visited);

        this.grad = 1;
        for(Value value : topo.reversed()){
            int adj = value.children.length;
            for(int i = 0; i < adj; i++){
                value.children[i].grad += value.grad * value.local_grads[i];
            }
        }
    }

    /**
     * Helper method for backward() that performs a topological sort of the computational graph. <br>
     * @param vertex - current node being visited
     * @param topo - topological ordering of the graph
     * @param visited - set of visited nodes
     */
    private void topoSort(Value vertex, List<Value> topo, Set<Value> visited){
        visited.add(vertex);
        for(Value child : vertex.children){
            if(!visited.contains(child)){
                topoSort(child, topo, visited);
            }
        }
        topo.add(vertex);
    }

    /**
     * Returns the numeric value stored in this Value object. <br>
     * @return - the numeric value stored in this Value object
     */
    public double asNum(){
        return data;
    }

    /**
     * Returns the gradient stored in this Value object. <br>
     * @return - the gradient stored in this Value object
     */
    public double getGrad(){
        return grad;
    }

    public void setGrad(double grad){
        this.grad = grad;
    }

    /**
     * Modifies the numeric value in-place without instantiating <code>Value</code>, preventing
     * gradient accumulation / unncessary nodes. 
     */
    public void changeData(double val){
        data += val;
    }
}
