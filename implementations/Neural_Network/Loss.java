public class Loss {
    public static double MSE(Matrix w, Matrix w_hat){
        int N = w.shape()[0];
        int O = w.shape()[1];
        double res = 0.0;
        for(int i = 0; i < N; i++){
            for(int j = 0; j < O; j++){
                res += Math.pow(w_hat.get(i,j) - w.get(i,j), 2);
            }
        }
        return res / (2.0 * N * O);
    }

    public static Matrix mseDerivative(Matrix w, Matrix w_hat, Matrix X) throws InvalidValue {
        int N = w.shape()[0];
        int O = w.shape()[1];
        int F = X.shape()[1];
        Matrix result = new Matrix(O, F);

        for(int o = 0; o < O; o++){
            for(int f = 0; f < F; f++){
                double sum = 0.0;
                for(int n = 0; n < N; n++){
                    double error = w_hat.get(n,o) - w.get(n,o);
                    sum += error * X.get(n,f);
                }
                result.set(o, f, sum / (N * O));
            }
        }
        return result;
    }

    public static double binaryCrossEntropyLoss(Matrix y, Matrix p){
        double res = 0.0;
        for(int i = 0; i < y.shape()[0]; i++){
            for(int j = 0; j < y.shape()[1]; j++){
                double pij = clip(p.get(i, j));
                res += -(((Math.log(pij)) * y.get(i, j)) + ((Math.log(1-pij)) * (1-y.get(i, j))));
            }
        }
        return res / y.shape()[0];
    }

    public static Matrix binaryCrossEntropyLossDerivative(Matrix y, Matrix p) throws InvalidValue{
        Matrix res = new Matrix(y.shape()[0], y.shape()[1]);
        for(int i = 0; i < y.shape()[0]; i++){
            for(int j = 0; j < y.shape()[1]; j++){
                double pij = clip(p.get(i, j));
                double val = (pij - y.get(i, j)) / (pij * (1-pij));
                res.set(i, j, val / y.shape()[0]);
            }
        }
        return res;
    }

    private static final double EPS = 1e-12;

    private static double clip(double val){
        return Math.min(Math.max(val, EPS), 1.0 - EPS);
    }

    public static double crossEntropyLoss(Matrix y, Matrix p){
        double res = 0.0;
        for(int i = 0; i < y.shape()[0]; i++){
            for(int j = 0; j < y.shape()[1]; j++){
                res += -(y.get(i, j) * Math.log(clip(p.get(i, j))));
            }
        }
        return res / y.shape()[0];
    }

    public static Matrix crossEntropyLossDerivative(Matrix y, Matrix p) throws InvalidValue{
        Matrix res = new Matrix(p);
        for(int i = 0; i < y.shape()[0]; i++){
            for(int j = 0; j < y.shape()[1]; j++){
                res.set(i,j,(y.get(i, j) / clip(p.get(i, j))));
            }
        }
        return new Matrix(Matrix.scalerMul(res, (-1.0 / y.shape()[0])));
    }

    public static Matrix softmaxCrossEntropyDerivative(Matrix y, Matrix p){
        Matrix res = new Matrix(p);
        for(int i = 0; i < y.shape()[0]; i++){
            for(int j = 0; j < y.shape()[1]; j++){
                res.set(i,j,(y.get(i, j) - p.get(i, j)));
            }
        }
        return new Matrix(Matrix.scalerMul(res, (1.0 / y.shape()[0])));
    }
}