package examples;

import java.io.IOException;

import core.Matrix;
import data.ModelParser;
import nn.module.Linear;
import nn.module.ModuleBase;

public class ModelIO {
    public static class Model extends ModuleBase{
        private Linear linear1, linear2;
        public Model(){
            linear1 = new Linear(784, 128); 
            linear2 = new Linear(128, 10);
            addRequirements();
        }

        public Matrix forward(Matrix input){
            return linear1.forward(input);
        }
    }

    public static void main(String[] args) throws IOException{
        Model model = new Model();
        ModelParser.loadModel(model, "examples/test");
    }
}
