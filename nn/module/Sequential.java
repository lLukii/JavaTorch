package nn.module;

import core.Matrix;

public class Sequential extends ModuleBase {
    private ModuleBase[] modules;
    public Sequential(ModuleBase... modules){
        this.modules = modules;
        addRequirements();
    }

    public Matrix forward(Matrix input){
        for(ModuleBase module : modules){
            input = module.forward(input);
        }
        return input;
    }
}
