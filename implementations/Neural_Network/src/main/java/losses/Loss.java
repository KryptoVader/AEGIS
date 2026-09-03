package losses;

import core.Matrix;
import exceptions.InvalidValue;

public interface Loss {
    double forward(Matrix y, Matrix prediction);
    Matrix derivative(Matrix y, Matrix prediction) throws InvalidValue;
}
