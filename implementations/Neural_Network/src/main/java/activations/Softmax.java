package activations;
import core.Matrix;
import exceptions.InvalidValue;

public class Softmax implements Activation {
    @Override
    public Matrix forward(Matrix z) throws InvalidValue {
        int N=z.shape()[0], O=z.shape()[1];
        Matrix result=new Matrix(N,O);

        for(int i=0;i<N;i++){
            double max=z.get(i,0);
            for(int j=1;j<O;j++){
                max=Math.max(max,z.get(i,j));
            }
            double sum=0.0;

            for(int j=0;j<O;j++){
                double exp=Math.exp(z.get(i,j)-max);
                result.set(i,j,exp);
                sum+=exp;
            }

            for(int j=0;j<O;j++){
                result.set(i,j,result.get(i,j)/sum);
            }
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix dLA, Matrix z) throws InvalidValue {
        Matrix a=forward(z);
        int N=z.shape()[0], O=z.shape()[1];
        Matrix dZ=new Matrix(N,O);

        for(int i=0;i<N;i++){
            for(int j=0;j<O;j++){
                double gradient=0.0;
                for(int k=0;k<O;k++){
                    double delta=(j==k)?1.0:0.0;
                    double jacobian=a.get(i,j)*(delta-a.get(i,k));
                    gradient+=dLA.get(i,k)*jacobian;
                }

                dZ.set(i,j,gradient);
            }
        }
        return dZ;
    }
}
