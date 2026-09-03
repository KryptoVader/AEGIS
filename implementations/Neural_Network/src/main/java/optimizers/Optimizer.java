package optimizers;

import core.Matrix;
import exceptions.InvalidValue;

public interface Optimizer {
    public Matrix update(Matrix param, Matrix grad) throws InvalidValue;
}
