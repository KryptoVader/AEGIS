package losses;

import core.Matrix;
import exceptions.InvalidValue;

public class MSE implements Loss {
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
