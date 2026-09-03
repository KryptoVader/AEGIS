package activations;

import core.Matrix;
import exceptions.InvalidValue;

public class Sigmoid implements Activation {
    @Override
    public Matrix forward(Matrix z) throws InvalidValue {
        int N=z.shape()[0], O=z.shape()[1];
        Matrix result=new Matrix(N,O);
        for(int i=0;i<N;i++){
            for(int j=0;j<O;j++){
                double value=z.get(i,j);
                double sigmoid=1.0/(1.0+Math.exp(-value));
                result.set(i,j,sigmoid);
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix dLA, Matrix z) throws InvalidValue {
        Matrix a=forward(z);
        int N=z.shape()[0], O=z.shape()[1];
        Matrix derivative=new Matrix(N,O);
        for(int i=0;i<N;i++){
            for(int j=0;j<O;j++){
                double p=a.get(i,j);
                derivative.set(i,j,p*(1.0-p));
            }
        }
        return Matrix.Hadamard(dLA,derivative);
    }
}
