package activations;
import core.Matrix;
import exceptions.InvalidValue;

public class ReLU implements Activation {
    @Override
    public Matrix forward(Matrix z) throws InvalidValue {
        int N=z.shape()[0], O=z.shape()[1];
        Matrix result=new Matrix(N,O);
        for(int i=0;i<N;i++){
            for(int j=0;j<O;j++){
                result.set(i,j,Math.max(0.0,z.get(i,j)));
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix dLA, Matrix z) throws InvalidValue {
        int N=z.shape()[0], O=z.shape()[1];
        Matrix derivative=new Matrix(N,O);
        for(int i=0;i<N;i++){
            for(int j=0;j<O;j++){
                derivative.set(i,j,z.get(i,j)>0.0?1.0:0.0);
            }
        }
        return Matrix.Hadamard(dLA,derivative);
    }
}