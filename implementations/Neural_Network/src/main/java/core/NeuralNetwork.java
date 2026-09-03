package core;

import java.util.Random;
import layers.Layer;
import layers.Linear;
import losses.Loss;
import losses.CrossEntropy;
import optimizers.Optimizer;
import optimizers.SGD;
import activations.ReLU;
import activations.Softmax;
import exceptions.InvalidValue;

public class NeuralNetwork {

    private Layer[] layers;
    private Loss loss;
    private Optimizer optimizer;

    public NeuralNetwork(Layer[] layers, Loss loss, Optimizer optimizer) {
        this.layers = layers;
        this.loss = loss;
        this.optimizer = optimizer;
    }

    public Matrix forward(Matrix X) throws InvalidValue {
        Matrix y_pred = this.layers[0].forward(X);
        for (int i = 1; i < this.layers.length; i++) {
            y_pred = this.layers[i].forward(y_pred);
        }
        return y_pred;
    }

    public void backward(Matrix dLA) throws InvalidValue {
        Matrix dA = dLA;
        for (int i = layers.length - 1; i >= 0; i--) {
            dA = layers[i].backward(dA);
        }
    }

    public double train(Matrix X, Matrix y) throws InvalidValue {
        Matrix pred = forward(X);
        double loss_val = this.loss.forward(y, pred);
        Matrix dLA = this.loss.derivative(y, pred);
        backward(dLA);
        for (Layer l : this.layers) {
            Linear li = l.getLinear();
            li.setW(this.optimizer.update(li.getW(), li.getDW()));
            li.setB(this.optimizer.update(li.getB(), li.getDB()));

        }
        return loss_val;
    }

    public double train(Matrix X, Matrix y, int batchSize) throws InvalidValue {
        int N = X.shape()[0];
        int[] indices = new int[N];

        for (int i = 0; i < N; i++) {
            indices[i] = i;
        }

        shuffle(indices);
        double totalLoss = 0.0;
        int batches = 0;

        for (int start = 0; start < N; start += batchSize) {
            int end = Math.min(start + batchSize, N);
            int[] batchIndices = new int[end - start];
            for (int i = 0; i < batchIndices.length; i++) {
                batchIndices[i] = indices[start + i];
            }
            Matrix XBatch = X.getRows(batchIndices);
            Matrix yBatch = y.getRows(batchIndices);
            totalLoss += train(XBatch, yBatch);
            batches++;
        }
        return totalLoss / batches;
    }

    private void shuffle(int[] indices) {
        Random random = new Random();
        for (int i = indices.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    public Matrix predict(Matrix X) throws InvalidValue {
        Matrix probabilities = forward(X);
        Matrix result = new Matrix(probabilities.shape()[0], 1);
        for (int i = 0; i < probabilities.shape()[0]; i++) {
            int predicted = 0;
            for (int j = 1; j < probabilities.shape()[1]; j++) {
                if (probabilities.get(i, j) > probabilities.get(i, predicted)) {
                    predicted = j;
                }
            }
            result.set(i, 0, predicted);
        }
        return result;
    }

    public double accuracy(Matrix X, Matrix y) throws InvalidValue {
        Matrix prediction = predict(X);
        int correct = 0;
        for (int i = 0; i < y.shape()[0]; i++) {
            int predictedClass = (int) prediction.get(i, 0);
            int actualClass = 0;
            for (int j = 0; j < y.shape()[1]; j++) {
                if (y.get(i, j) == 1.0) {
                    actualClass = j;
                    break;
                }
            }
            if (predictedClass == actualClass) {
                correct++;
            }
        }
        return (double) correct / y.shape()[0];
    }

    public static void main(String[] args) {
        try {
            Layer l1 = new Layer(2, 4, new ReLU());
            Layer l2 = new Layer(4, 2, new Softmax());

            Matrix X = new Matrix(new double[][] {
                    { 0.0, 0.0 },
                    { 0.0, 1.0 },
                    { 1.0, 0.0 },
                    { 1.0, 1.0 }
            });

            Matrix Y = new Matrix(new double[][] {
                    { 1.0, 0.0 },
                    { 0.0, 1.0 },
                    { 0.0, 1.0 },
                    { 1.0, 0.0 }
            });

            NeuralNetwork model = new NeuralNetwork(
                    new Layer[] { l1, l2 },
                    new CrossEntropy(),
                    new SGD(0.1));

            for (int i = 0; i < 10000; i++) {
                double loss = model.train(X, Y, 2);

                if (i % 1000 == 0) {
                    System.out.println("Epoch: " + i + " Loss: " + loss);
                }
            }

            System.out.println("Probabilities:");
            System.out.println(model.forward(X));

            System.out.println("Predictions:");
            System.out.println(model.predict(X));
            System.out.println("Accuracy: " + model.accuracy(X, Y));

        } catch (InvalidValue e) {
            System.err.println(e.getMessage());
        }
    }
}
