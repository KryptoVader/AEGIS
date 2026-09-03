package metrics;

import core.Matrix;
import exceptions.InvalidValue;

public class Metrics {
    public static Matrix confusionMatrix(Matrix y, Matrix pred) throws InvalidValue {
        int classes = y.shape()[1];
        Matrix res = new Matrix(classes);
        for (int i = 0; i < y.shape()[0]; i++) {
            int actual = argmax(y.getRow(i));
            int predicted = argmax(pred.getRow(i));
            res.set(actual, predicted, res.get(actual, predicted) + 1);
        }
        return res;
    }

    private static int argmax(double[] row) {
        int index = 0;
        for (int i = 1; i < row.length; i++) {
            if (row[i] > row[index]) {
                index = i;
            }
        }
        return index;
    }
}
