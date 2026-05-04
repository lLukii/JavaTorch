package core;

/**
 * Base matrix class for all matrix operations. Provides basic functionalities like addition, subtraction, multiplication, scalar multiplication, and transposition.
 * 
 * @author lLukii
 */
public class Matrix{
    private Value[][] data;
    private int rows, cols;

    /**
     * Creates a matrix of given dimensions and fills it with the specified value.
     * @param rows - number of rows
     * @param cols - number of columns
     * @param val - value to fill the matrix with
     */
    public Matrix(int rows, int cols, Value val){
        data = new Value[rows][cols];
        this.rows = rows;
        this.cols = cols;
        fill(val);
    }

    public Matrix(int rows, int cols){
        this(rows, cols, new Value(0));
    }

    /**
     * Fills a submatrix defined by the specified row and column ranges with the given value.
     * @param startRow - starting row index
     * @param endRow - ending row index
     * @param startCol - starting column index
     * @param endCol - ending column index
     * @param val - value to fill the submatrix with
     */
    public void fill(int startRow, int endRow, int startCol, int endCol, Value val){
        for(int i = startRow; i <= endRow; i++){
            for(int j = startCol; j <= endCol; j++){
                data[i][j] = val;
            }
        }
    }

    /**
     * Fills the entire matrix with the specified value.
     * @param val
     */
    public void fill(Value val){
        fill(0, rows - 1, 0, cols - 1, val);
    }

    public int getRows(){return rows;}
    public int getCols(){return cols;}

    public Value get(int i, int j){return data[i][j];}
    public Matrix get(int startRow, int endRow, int startCol, int endCol){
        Matrix sub = new Matrix(endRow - startRow + 1, endCol - startCol + 1);
        for(int i = startRow; i <= endRow; i++){
            for(int j = startCol; j <= endCol; j++){
                sub.set(i - startRow, j - startCol, data[i][j]);
            }
        }
        return sub;
    }

    public void set(int i, int j, Value val){data[i][j] = val;}
    
    /**
     * Adds two matrices element-wise and returns the resulting matrix.
     * @param a
     * @param b
     * @return
     */
    public static Matrix add(Matrix a, Matrix b){
        Matrix sum = new Matrix(a.rows, a.cols);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++){
                sum.set(i, j, a.get(i, j).add(b.get(i, j)));
            }
        }
        return sum;
    }

    /**
     * Subtracts the second matrix from the first matrix element-wise and returns the resulting matrix.
     * @param a
     * @param b
     * @return
     */
    public static Matrix subtract(Matrix a, Matrix b){
        Matrix diff = new Matrix(a.rows, a.cols);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++){
                diff.set(i, j, a.get(i, j).sub(b.get(i, j)));
            }
        }
        return diff;
    }

    public Matrix add(Matrix b){
        return Matrix.add(this, b);
    }

    /**
     * Multiplies two matrices using standard matrix multiplication and returns the resulting matrix.
     * @param a
     * @param b
     * @return
     */
    public static Matrix matMul(Matrix a, Matrix b){
        Matrix product = new Matrix(a.rows, b.cols);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < b.cols; j++){
                Value sum = new Value(0);
                for(int k = 0; k < a.cols; k++){
                    sum = sum.add(a.get(i, k).mul(b.get(k, j)));
                }
                product.set(i, j, sum);
            }
        }
        return product;
    }

    public Matrix matMul(Matrix b){
        return Matrix.matMul(this, b);
    }

    /**
     * Multiplies each element of the matrix by a scalar value and returns the resulting matrix.
     * @param a
     * @param scalar
     * @return
     */
    public static Matrix scalarMul(Matrix a, Value scalar){
        Matrix product = new Matrix(a.rows, a.cols);
        for(int i = 0; i < a.rows; i++){
            for(int j = 0; j < a.cols; j++){
                product.set(i, j, a.get(i, j).mul(scalar));
            }
        }
        return product;
    }

    public Matrix scalarMul(Value scalar){
        return Matrix.scalarMul(this, scalar);
    }

    /**
     * Transposes the matrix by swapping its rows and columns, returning a new transposed matrix.
     * @return
     */
    public Matrix transpose(){
        Matrix transposed = new Matrix(cols, rows);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                transposed.set(j, i, data[i][j]);
            }
        }
        return transposed;
    }

    public String toString(){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < rows; i++){
            sb.append("[");
            for(int j = 0; j < cols; j++){
                sb.append(String.format("%.2f", data[i][j].asNum()));
                if(j < cols - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}