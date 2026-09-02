public interface Optimizer {
    public Matrix update(Matrix param, Matrix grad) throws InvalidValue;
}


class SGD implements Optimizer{
    private double lr;

    public SGD(double lr){
        this.lr = lr;
    }

    @Override
    public Matrix update(Matrix param, Matrix grad) throws InvalidValue{
        Matrix x = Matrix.sub(param, new Matrix(Matrix.scalerMul(grad, lr)));
        return x;
    }
}