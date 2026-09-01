public class Loss {
    public static double MSE(Matrix w, Matrix w_hat){
        double res = 0.0;
        for(int i = 0; i < w.shape()[1]; i++){
            res += Math.pow((w_hat.get(0,i)- w.get(0,i)),2);
        }
        return res / (w.shape()[1] * 2);
    }

    public static Matrix mseDerivative(Matrix w, Matrix w_hat, Matrix X) throws InvalidValue{
        Matrix diff = new Matrix(w.shape()[0], w.shape()[1]);
        for(int i = 0; i < w.shape()[1]; i++){
            diff.set(0,i ,(w_hat.get(0,i)- w.get(0,i)));
        }
        Matrix res = new Matrix(X);
        for(int i = 0; i < res.shape()[0]; i++){
            for (int j = 0; j < res.shape()[1]; j++){
                res.set(i, j, ((X.get(i,j) * diff.get(0,i)) / w.shape()[1]));
            }
        }
        Matrix result = new Matrix(1, X.shape()[1]);
        for(int i = 0; i < X.shape()[1]; i++){
            double sum = 0;
            for(double ele: res.getCol(i)){
                sum += ele;
            }
            result.set(0,i, sum);
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
    public static void main(String[] args) {
        try {
            Matrix y = new Matrix(3, 4);
            y.set(0, 0, 1.0);
            y.set(1, 2, 1.0); 
            y.set(2, 1, 1.0);

            Matrix p = new Matrix(3, 4);
            p.set(0, 0, 0.7);
            p.set(0, 1, 0.1);
            p.set(0, 2, 0.1);
            p.set(0, 3, 0.1);

            p.set(1, 0, 0.1);
            p.set(1, 1, 0.2);
            p.set(1, 2, 0.6);
            p.set(1, 3, 0.1);

            p.set(2, 0, 0.1);
            p.set(2, 1, 0.5);
            p.set(2, 2, 0.2);
            p.set(2, 3, 0.2);

            System.out.println("y:");
            System.out.println(y);
            System.out.println("\np:");
            System.out.println(p);

            Matrix derivative = Loss.softmaxCrossEntropyDerivative(y, p);

            System.out.println("\nSoftmax + CE derivative:");
            System.out.println(derivative);
            System.out.println("\nRow sums:");

            for (int i = 0; i < derivative.shape()[0]; i++) {
                double sum = 0.0;
                for (int j = 0; j < derivative.shape()[1]; j++) {
                    sum += derivative.get(i, j);
                }
                System.out.println("Row " + i + ": " + sum);
            }
        } catch (InvalidValue e) {
            System.out.println(e.getMessage());
        }
    }
}