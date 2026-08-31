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

    public static void main(String[] args) {
    try {
        Matrix m = new Matrix(1, 5);

        m.set(0, 0, -10);
        m.set(0, 1, -1);
        m.set(0, 2, 0);
        m.set(0, 3, 1);
        m.set(0, 4, 10);

        System.out.println("Input:               " + m);
        System.out.println("ReLU:                " + Activation.ReLU(m));
        System.out.println("ReLU Derivative:     " + Activation.reluDerivative(m));
    } catch (InvalidValue e) {
        System.out.println(e.getMessage());
    }
}
}
