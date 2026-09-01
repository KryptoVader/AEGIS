public class LogisticRegression{
    private Matrix w;
    private double b;
    private double lr;
    private int epochs;

    public LogisticRegression(int features, double lr, int epochs) throws InvalidValue{
        this.w = new Matrix(1, features);
        this.b = 0.0;
        this.lr = lr;
        this.epochs = epochs;
    }

    public void fit(Matrix X, Matrix y) throws InvalidValue{
        for(int epoch = 0; epoch < this.epochs; epoch++){
            for(int i = 0; i < X.shape()[0]; i++){
                double z = Matrix.dot(X.getRow(i), Matrix.Transpose(this.w).getCol(0)) + this.b;
                double p_z = (1.0 / (1.0 + Math.exp(-z)));

                double grad = y.get(i,0) - p_z;
                this.w = Matrix.add(this.w, new Matrix(Matrix.scalerMul(new Matrix(X.getRow(i)), this.lr*grad)));
                this.b = this.b + this.lr*grad;
            }
        }
    }

    public Matrix predict(Matrix X) throws InvalidValue {
        return predict(X, 0.5);
    }

    public Matrix predict(Matrix X, double thres) throws InvalidValue{
        Matrix y_hat = new Matrix(X.shape()[0], 1);
        for(int i = 0; i < X.shape()[0]; i++){
            double z = Matrix.dot(X.getRow(i), Matrix.Transpose(this.w).getCol(0)) + this.b;
            double p_z = (1.0 / (1.0 + Math.exp(-z)));
            y_hat.set(i,0,p_z >= thres ? 1.0 : 0.0);
        }
        return y_hat;
    }
    
    public Matrix coef(){
        return new Matrix(this.w);
    }

    public double intercept(){
        return this.b;
    }

}
