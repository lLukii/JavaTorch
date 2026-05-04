package nn.module;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.Matrix;
import core.Parameter;
import core.Value;

/**
 * Base class for neural network modules. 
 */
public abstract class ModuleBase {
    Map<String, Parameter> stateDict = new HashMap<>();
    Map<String, ModuleBase> subModules = new HashMap<>();
    
    public ModuleBase(){}

    public void addRequirements(){
        Field[] fields = this.getClass().getDeclaredFields();
        for(Field field : fields){
            field.setAccessible(true);
            try{
                Object t = field.get(this);
                if(t instanceof Parameter){
                    stateDict.put(field.getName(), (Parameter) t);
                }
                else if(t instanceof ModuleBase){
                    subModules.put(field.getName(), (ModuleBase) t);
                }
            }
            catch(IllegalAccessException e){
                System.out.println("Error accessing attribute " + field.getName());
            }
        }
    }

    /**
     * The required forward method defines the computation performed at every call during inference. It takes an input matrix and produces an output matrix based on the specific module's functionality.
     * @param input - the input matrix to the module
     * @return the output matrix resulting from the module's computation on the input
     */
    public abstract Matrix forward(Matrix input);

    /**
     * Returns a map corresponding to trainable parameters in the current module. 
     * @return a map of parameter names to their corresponding Parameter objects contained within the current module
     */
    public Map<String, Parameter> getStateDict(){
        return stateDict;
    }    

    /**
     * Returns a map corresponding to submodules contained within the current module.
     * @return a map of submodule names to their corresponding ModuleBase objects contained within the current module
     */
    public Map<String, ModuleBase> getSubModules(){
        return subModules; 
    }

    /**
     * Recursively searches through all modules/submodules and compiles trainable parameters into a single list. 
     * Required for optimizer algorithms that cache previous gradients to improve convergence. 
     * @return flattened list of trainable parameters in the neural network. 
     */
    public List<Value> parameters(){
        if(stateDict.size() == 0){
            return new ArrayList<>();
        }
        List<Value> params = new ArrayList<>();
        for(Matrix param : stateDict.values()){
            for(int i = 0; i < param.getRows(); i++){
                for(int j = 0; j < param.getCols(); j++){
                    params.add(param.get(i, j));
                }
            }
        }
        for(ModuleBase subModule : subModules.values()){
            params.addAll(subModule.parameters());
        }
        return params;
    }
}
