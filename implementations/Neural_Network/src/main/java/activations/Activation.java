package activations;
import core.Matrix;
import exceptions.InvalidValue;

public interface Activation {
    Matrix forward(Matrix z) throws InvalidValue;
    Matrix backward(Matrix dLA, Matrix z) throws InvalidValue;
}
