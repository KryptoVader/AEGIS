public class Layer {
    private Linear l;
    private Activation a;
    private Matrix z;

    public Layer(int input_features, int output_features, Activation a) throws InvalidValue {
        this.l = new Linear(input_features, output_features);
        this.a = a;
    }

    public Layer(int input_features, int output_features) throws InvalidValue {
        this.l = new Linear(input_features, output_features);
        this.a = null;
    }

    public Matrix forward(Matrix X) throws InvalidValue {
        Matrix z = l.forward(X);
        this.z = z;
        if (this.a != null) {
            Matrix pred = a.forward(z);
            return new Matrix(pred);
        }
        return new Matrix(z);
    }

    public Matrix backward(Matrix dLA) throws InvalidValue {
        Matrix dz = dLA;
        if (this.a != null) {
            dz = Matrix.Hadamard(dLA, this.a.derivative(this.z));
        }

        Matrix dX = this.l.backward(dz);
        return dX;
    }

    public Linear getLinear(){
        return this.l;
    }
}
