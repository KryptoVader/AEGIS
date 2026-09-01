import java.util.function.DoubleUnaryOperator;
public interface Activation {
    Matrix forward(Matrix m) throws InvalidValue;
    Matrix derivative(Matrix m) throws InvalidValue;
}

class ReLU implements Activation {
    @Override
    public Matrix forward(Matrix m) throws InvalidValue {
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for (int i = 0; i < m.shape()[0]; i++) {
            for (int j = 0; j < m.shape()[1]; j++) {
                double val = m.get(i, j) > 0 ? m.get(i, j) : 0.0;
                res.set(i, j, val);
            }
        }
        return res;
    }

    @Override
    public Matrix derivative(Matrix m) throws InvalidValue {
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for (int i = 0; i < m.shape()[0]; i++) {
            for (int j = 0; j < m.shape()[1]; j++) {
                double val = m.get(i, j) > 0 ? 1.0 : 0.0;
                res.set(i, j, val);
            }
        }
        return res;
    }
}

class Sigmoid implements Activation {
    @Override
    public Matrix forward(Matrix m) throws InvalidValue {
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        for (int i = 0; i < m.shape()[0]; i++) {
            for (int j = 0; j < m.shape()[1]; j++) {
                double val = 1.0 / (1.0 + Math.exp(-m.get(i, j)));
                res.set(i, j, val);
            }
        }
        return res;
    }


    @Override
    public Matrix derivative(Matrix m) throws InvalidValue {
        DoubleUnaryOperator deri = x -> x * (1 - x);
        Matrix res = new Matrix(m.shape()[0], m.shape()[1]);
        Matrix sig = forward(m);

        for (int i = 0; i < m.shape()[0]; i++) {
            for (int j = 0; j < m.shape()[1]; j++) {
                double val = sig.get(i, j);
                res.set(i, j, deri.applyAsDouble(val));
            }
        }
        return res;
    }
}

class Softmax implements Activation {
    private double sum(double[] r) {
        double res = 0.0;
        double maxValue = max(r);

        for (double ele : r){
            res += Math.exp(ele - maxValue);
        }
        return res;
    }

    private double max(double[] r) {
        double res = Double.NEGATIVE_INFINITY;
        for (double ele : r) {
            if (ele > res) {
                res = ele;
            }
        }
        return res;
    }


    @Override
    public Matrix forward(Matrix z) {
        Matrix res = new Matrix(z);

        for (int i = 0; i < z.shape()[0]; i++) {
            double[] row = z.getRow(i);
            double maxValue = max(row);
            double denominator = sum(row);
            for (int j = 0; j < z.shape()[1]; j++) {
                double val = Math.exp(z.get(i, j) - maxValue) / denominator;
                res.set(i, j, val);
            }
        }
        return res;
    }

    @Override
    public Matrix derivative(Matrix z) throws InvalidValue {
        if (z.shape()[0] != 1) {
            throw new InvalidValue(
                "Softmax derivative requires a single row."
            );
        }

        Matrix softmax = forward(z);
        Matrix I = Matrix.identity(z.shape()[1]);
        Matrix p = new Matrix(softmax.getRow(0));
        Matrix arr = Matrix.sub(I,Matrix.matmul(Matrix.ones(softmax.shape()[1],1),p));
        Matrix res = new Matrix(softmax.shape()[1],softmax.shape()[1]);

        for (int j = 0; j < arr.shape()[0]; j++) {
            for (int k = 0; k < arr.shape()[1]; k++) {
                res.set(j,k,p.get(0, j) * arr.get(j, k));
            }
        }
        return res;
    }
}