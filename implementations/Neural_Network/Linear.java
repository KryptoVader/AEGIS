public class Linear {
    private Matrix x;
    private Matrix w;
    private Matrix b;
    private Matrix dW;
    private Matrix dB;
    public Linear(int inputFeatures, int outputFeatures) throws InvalidValue{
        this.w = new Matrix(outputFeatures, inputFeatures);
        this.b = new Matrix(1,outputFeatures);
    }

    public Linear(Matrix w, Matrix b) throws InvalidValue {
        this.w = new Matrix(w);
        this.b = new Matrix(b);
    }

    public void setWeight(int i, int j, double value) {
        this.w.set(i, j, value);
    }

    public void setBias(int j, double value) {
        this.b.set(0, j, value);
    }

    public Matrix forward(Matrix X) throws InvalidValue{
        this.x = X;
        Matrix xwt = Matrix.matmul(X, Matrix.Transpose((this.w)));
        for(int i = 0; i < xwt.shape()[0]; i++){
            for(int j = 0; j < xwt.shape()[1]; j++){
                xwt.set(i,j, xwt.get(i,j) + this.b.get(0,j));
            }
        }
        return xwt;
    }

    private double sum(double[] arr){
        double res = 0.0;
        for(double ele: arr){
            res += ele;
        }
        return res;
    }

    public Matrix backward(Matrix dz) throws InvalidValue{
        this.dW = Matrix.matmul(Matrix.Transpose(dz),this.x);
        this.dB = new Matrix(1, dz.shape()[1]);
        for(int i = 0; i < dz.shape()[1]; i++){
            this.dB.set(0, i, sum(dz.getCol(i)));
        }
        return Matrix.matmul(dz, this.w);
    }

    public Matrix getDW(){
        return dW;
    }

    public Matrix getDB(){
        return dB;
    }
}
