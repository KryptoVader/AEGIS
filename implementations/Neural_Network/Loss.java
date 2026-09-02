public interface Loss {
    double forward(Matrix y, Matrix prediction);
    Matrix derivative(Matrix y, Matrix prediction) throws InvalidValue;
}


class MSE implements Loss {
    @Override
    public double forward(Matrix y, Matrix prediction) {
        int N = y.shape()[0];
        int O = y.shape()[1];
        double result = 0.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < O; j++) {
                double error = prediction.get(i, j) - y.get(i, j);
                result += error * error;
            }
        }
        return result / (2.0 * N * O);
    }

    @Override
    public Matrix derivative(Matrix y, Matrix prediction) throws InvalidValue {
        int N = y.shape()[0];
        int O = y.shape()[1];
        Matrix result = new Matrix(N, O);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < O; j++) {
                double error = prediction.get(i, j) - y.get(i, j);
                result.set(i, j, error / (N * O));
            }
        }
        return result;
    }
}


class CrossEntropy implements Loss {
    private static final double EPS = 1e-12;

    @Override
    public double forward(Matrix y, Matrix prediction) {
        int N = y.shape()[0];
        double result = 0.0;

        for (int i = 0; i < y.shape()[0]; i++) {
            for (int j = 0; j < y.shape()[1]; j++) {
                double p = clip(prediction.get(i, j));
                result += -(y.get(i, j) * Math.log(p));
            }
        }
        return result / N;
    }

    @Override
    public Matrix derivative(Matrix y, Matrix prediction) throws InvalidValue {
        int N = y.shape()[0];
        int O = y.shape()[1];
        Matrix result = new Matrix(N, O);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < O; j++) {
                double p = clip(prediction.get(i, j));
                result.set(i, j, -y.get(i, j) / (p * N));
            }
        }
        return result;
    }

    private static double clip(double value) {
        return Math.min(Math.max(value, EPS), 1.0 - EPS);
    }
}