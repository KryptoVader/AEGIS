public class Perceptron {
    private Matrix w;
    private double b;
    private int epochs;
    public Perceptron(int i, int epochs) throws InvalidValue{
        this.w = new Matrix(1, i);
        this.b = 0.0;
        this.epochs = epochs;
    }

    public void fit(Matrix X, Matrix y) throws InvalidValue{
        for(int i = 0; i < this.epochs; i++){
            for(int j = 0; j < X.shape()[0]; j++){
                var xwt = Matrix.dot(X.getRow(j), Matrix.Transpose(this.w).getCol(0)) + this.b;
                double y_hat = xwt >= 0? 1.0 : -1.0;

                if (y_hat != y.get(j,0)){
                    this.w = Matrix.add(this.w, new Matrix(Matrix.scalerMul(new Matrix(X.getRow(j)), y.get(j, 0))));
                    this.b += y.get(j,0);
                }
            }
        }
    }

    public Matrix predict(Matrix X) throws InvalidValue{
        Matrix y_hat = new Matrix(X.shape()[0], 1);
        for(int j = 0; j < X.shape()[0]; j++){
            var xwt = Matrix.dot(X.getRow(j), Matrix.Transpose(this.w).getCol(0)) + this.b;
            y_hat.set(j,0, xwt >= 0? 1.0 : -1.0);
        }
        return y_hat;
    }

    public Matrix coef(){
        return new Matrix(this.w);
    }

    public double intercept(){
        return this.b;
    }

    public static void main(String[] args) {
        try {
            Matrix X = new Matrix(4, 2);
            X.set(0, 0, 2.0);
            X.set(0, 1, 2.0);
            X.set(1, 0, 3.0);
            X.set(1, 1, 1.0);
            X.set(2, 0, 0.0);
            X.set(2, 1, 0.0);
            X.set(3, 0, 1.0);
            X.set(3, 1, 0.0);

            Matrix y = new Matrix(4, 1);
            y.set(0, 0, 1.0);
            y.set(1, 0, 1.0);
            y.set(2, 0, -1.0);
            y.set(3, 0, -1.0);
            
            Perceptron p = new Perceptron(2, 10);
            System.out.println("X:");
            System.out.println(X);
            System.out.println("\ny:");
            System.out.println(y);
            System.out.println("\nBefore training:");
            System.out.println("Weights:");
            System.out.println(p.coef());
            System.out.println("Bias:");
            System.out.println(p.intercept());
            System.out.println("Predictions:");
            System.out.println(p.predict(X));

            p.fit(X, y);

            System.out.println("\nAfter training:");
            System.out.println("Weights:");
            System.out.println(p.coef());
            System.out.println("Bias:");
            System.out.println(p.intercept());
            System.out.println("Predictions:");
            System.out.println(p.predict(X));

            Matrix predictions = p.predict(X);
            boolean correct = true;

            for (int i = 0; i < y.shape()[0]; i++) {
                if (predictions.get(i, 0) != y.get(i, 0)) {
                    correct = false;
                    break;
                }
            }

            System.out.println("\nAll predictions correct: " + correct);
        } catch (InvalidValue e) {

            System.out.println(e.getMessage());
        }
    }
}