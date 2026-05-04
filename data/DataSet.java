package data;

import core.Matrix;

/**
 * Abstract class representing a dataset.
 * @author lLukii
 */
public abstract class DataSet {
    public DataSet(){}

    /**
     * Returns the input and target Value arrays for the given index. <br>
     * @param index
     * @return an array of two Value arrays: [input, target]
     */
    public abstract Matrix[] getItem(int index); 

    /**
     * Returns the size of the dataset
     * @return - the size of the dataset
     */
    public abstract int getSize();
}
