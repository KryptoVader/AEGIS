import java.util.function.DoubleUnaryOperator;
public class Activation {
    public static Matrix Sigmoid(Matrix m) throws InvalidValue{
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for(int i = 0; i < m.shape()[0]; i++){
            for (int j = 0; j < m.shape()[1]; j++){
                double val = 1.0 / (1.0 + Math.exp(- m.get(i, j)));
                res.set(i, j, val);
            }
        }
        return res;
    }

    public static Matrix sigmoidDerivative(Matrix m) throws InvalidValue{
        DoubleUnaryOperator deri = x -> x * (1 - x);
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        Matrix sig = Sigmoid(m);
        for(int i = 0; i < m.shape()[0]; i++){
            for (int j = 0; j < m.shape()[1]; j++){
                double val = sig.get(i, j);
                res.set(i, j, deri.applyAsDouble(val));
            }
        }
        return res;
    }

    public static Matrix ReLU(Matrix m) throws InvalidValue{
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for(int i = 0; i < m.shape()[0]; i++){
            for (int j = 0; j < m.shape()[1]; j++){
                double val = m.get(i, j) > 0 ? m.get(i, j) : 0.0;
                res.set(i, j, val);
            }
        }
        return res;
    }

    public static Matrix reluDerivative(Matrix m) throws InvalidValue{
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for(int i = 0; i < m.shape()[0]; i++){
            for (int j = 0; j < m.shape()[1]; j++){
                double val = m.get(i, j) > 0 ? 1.0 : 0.0;
                res.set(i, j, val);
            }
        }
        return res;
    }

    private static double sum(double[] r){
        double res = 0.0;
        for(double ele: r){
            res += Math.exp(ele - max(r));
        }
        return res;
    }

    private static double max(double[] r){
        double res = Double.NEGATIVE_INFINITY;
        for(double ele : r){
            if(ele > res)
                res = ele;
        }
        return res;
    }

    public static Matrix Softmax(Matrix z){
        Matrix res = new Matrix(z);
        for(int i = 0; i < z.shape()[0]; i++){
            for(int j = 0; j < z.shape()[1]; j++){
                res.set(i, j, (Math.exp(z.get(i, j) - max(z.getRow(i))) / sum(z.getRow(i))));
            }
        }
        return res;
    }

    public static Matrix softmaxDerivative(Matrix z, int i, Matrix softmax) throws InvalidValue{
        Matrix I = Matrix.identity(z.shape()[1]);
        Matrix res = new Matrix(softmax.shape()[1],softmax.shape()[1]);
        Matrix p = new Matrix(softmax.getRow(i));
        Matrix arr = Matrix.sub(I,Matrix.matmul(Matrix.ones(softmax.shape()[1], 1),p));
        for (int j = 0; j < arr.shape()[0]; j++) {
            for (int k = 0; k < arr.shape()[1]; k++) {
                res.set(j,k,p.get(0, j) * arr.get(j, k));
            }
        }
        return res;
    }

    public static void main(String[] args){
        try {
            Matrix m = new Matrix(4, 3);
            m.set(0, 0, 2.0);
            m.set(0, 1, 1.0);
            m.set(0, 2, 0.0);

            m.set(1, 0, 1.0);
            m.set(1, 1, 2.0);
            m.set(1, 2, 3.0);
            m.set(2, 0, 5.0);
            m.set(2, 1, 5.0);
            m.set(2, 2, 5.0);
            m.set(3, 0, 1000.0);
            m.set(3, 1, 1001.0);
            m.set(3, 2, 1002.0);

            Matrix result = Activation.Softmax(m);

            System.out.println("Input:");
            System.out.println(m);
            System.out.println("\nSoftmax:");
            System.out.println(result);
            System.out.println("\nRow sums:");

            for (int i = 0; i < result.shape()[0]; i++) {
                double sum = 0.0;
                for (int j = 0; j < result.shape()[1]; j++) {
                    sum += result.get(i, j);
                }
                System.out.println("Row " + i + ": " + sum);
            }

        } catch (InvalidValue e) {
            System.out.println(e.getMessage());
        }
    }
}
