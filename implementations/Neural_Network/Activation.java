public interface Activation {
    Matrix forward(Matrix z) throws InvalidValue;
    Matrix backward(Matrix dLA, Matrix z) throws InvalidValue;
}

class ReLU implements Activation {
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

class Sigmoid implements Activation {
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

class Softmax implements Activation {
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