package examples;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.SwingWrapper;

import core.Matrix;
import core.NeuralNetwork;
import layers.Layer;
import activations.ReLU;
import activations.Softmax;
import losses.CrossEntropy;
import optimizers.SGD;
import exceptions.InvalidValue;

public class MNIST{
    private Matrix XTrain;
    private Matrix YTrain;
    private Matrix XTest;
    private Matrix YTest;

    public MNIST(String path)throws InvalidValue{
        try{
            XTrain=loadImages(path+"\\train-images-idx3-ubyte");
            YTrain=loadLabels(path+"\\train-labels-idx1-ubyte");
            XTest=loadImages(path+"\\t10k-images-idx3-ubyte");
            YTest=loadLabels(path+"\\t10k-labels-idx1-ubyte");
        }catch(IOException e){
            throw new InvalidValue("Could not load MNIST: "+e.getMessage());
        }
    }

    private Matrix loadImages(String path)throws IOException,InvalidValue{
        DataInputStream in=new DataInputStream(new FileInputStream(path));
        int magic=in.readInt();
        int N=in.readInt();
        int rows=in.readInt();
        int cols=in.readInt();

        if(magic!=2051){
            in.close();
            throw new InvalidValue("Invalid MNIST image file");
        }

        if(rows!=28||cols!=28){
            in.close();
            throw new InvalidValue("Expected 28x28 images");
        }

        double[][] data=new double[N][784];

        for(int i=0;i<N;i++){
            for(int j=0;j<784;j++){
                data[i][j]=in.readUnsignedByte()/255.0;
            }
        }

        in.close();
        return new Matrix(data);
    }

    private Matrix loadLabels(String path)throws IOException,InvalidValue{
        DataInputStream in=new DataInputStream(new FileInputStream(path));
        int magic=in.readInt();
        int N=in.readInt();

        if(magic!=2049){
            in.close();
            throw new InvalidValue("Invalid MNIST label file");
        }

        double[][] data=new double[N][10];

        for(int i=0;i<N;i++){
            int label=in.readUnsignedByte();

            if(label<0||label>9){
                in.close();
                throw new InvalidValue("Invalid MNIST label");
            }

            data[i][label]=1.0;
        }

        in.close();
        return new Matrix(data);
    }

    public Matrix getXTrain(){
        return XTrain;
    }

    public Matrix getYTrain(){
        return YTrain;
    }

    public Matrix getXTest(){
        return XTest;
    }

    public Matrix getYTest(){
        return YTest;
    }

    public static void main(String[] args){
        try{
            String path = (args.length > 0) ? args[0] : "C:\\AEGIS\\roadmap\\Deep_Learning\\mnist\\data\\MNIST\\raw";

            MNIST data=new MNIST(path);

            Matrix XTrain=data.getXTrain();
            Matrix YTrain=data.getYTrain();
            Matrix XTest=data.getXTest();
            Matrix YTest=data.getYTest();

            System.out.println("XTrain: "+XTrain.shape()[0]+" x "+XTrain.shape()[1]);
            System.out.println("YTrain: "+YTrain.shape()[0]+" x "+YTrain.shape()[1]);
            System.out.println("XTest: "+XTest.shape()[0]+" x "+XTest.shape()[1]);
            System.out.println("YTest: "+YTest.shape()[0]+" x "+YTest.shape()[1]);

            Layer l1=new Layer(784,128,new ReLU());
            Layer l2=new Layer(128,10,new Softmax());

            NeuralNetwork model=new NeuralNetwork(
                new Layer[]{l1,l2},
                new CrossEntropy(),
                new SGD(0.01)
            );

            int epochs=10;
            int batchSize=64;

            double[] epochValues=new double[epochs];
            double[] losses=new double[epochs];
            double[] trainAccuracies=new double[epochs];
            double[] testAccuracies=new double[epochs];

            for(int i=0;i<epochs;i++){
                double loss=model.train(XTrain,YTrain,batchSize);

                Matrix trainPrediction=model.forward(XTrain);
                Matrix testPrediction=model.forward(XTest);

                double trainAccuracy=accuracy(trainPrediction,YTrain);
                double testAccuracy=accuracy(testPrediction,YTest);

                epochValues[i]=i+1;
                losses[i]=loss;
                trainAccuracies[i]=trainAccuracy;
                testAccuracies[i]=testAccuracy;

                System.out.println(
                    "Epoch: "+(i+1)+
                    " Loss: "+loss+
                    " Train Accuracy: "+trainAccuracy+
                    " Test Accuracy: "+testAccuracy
                );
            }

            Matrix testPrediction=model.forward(XTest);

            int[][] confusion=confusionMatrix(YTest,testPrediction);

            System.out.println("\nConfusion Matrix:");

            System.out.print("     ");
            for(int i=0;i<10;i++){
                System.out.printf("%6d",i);
            }
            System.out.println();

            for(int i=0;i<10;i++){
                System.out.printf("%3d: ",i);

                for(int j=0;j<10;j++){
                    System.out.printf("%6d",confusion[i][j]);
                }

                System.out.println();
            }

            plotLoss(epochValues,losses);
            plotAccuracy(epochValues,trainAccuracies,testAccuracies);

        }catch(InvalidValue e){
            System.err.println(e.getMessage());
        }
    }

    private static double accuracy(Matrix prediction,Matrix y)throws InvalidValue{
        int correct=0;
        int N=y.shape()[0];

        for(int i=0;i<N;i++){
            int predicted=argmax(prediction.getRow(i));
            int actual=argmax(y.getRow(i));

            if(predicted==actual){
                correct++;
            }
        }

        return (double)correct/N;
    }

    private static int[][] confusionMatrix(Matrix y,Matrix prediction)throws InvalidValue{
        int[][] result=new int[10][10];

        for(int i=0;i<y.shape()[0];i++){
            int actual=argmax(y.getRow(i));
            int predicted=argmax(prediction.getRow(i));

            result[actual][predicted]++;
        }

        return result;
    }

    private static int argmax(double[] row){
        int index=0;

        for(int i=1;i<row.length;i++){
            if(row[i]>row[index]){
                index=i;
            }
        }

        return index;
    }

    private static void plotLoss(double[] epochs,double[] losses){
        XYChart chart=new XYChartBuilder()
            .width(900)
            .height(600)
            .title("MNIST Training Loss")
            .xAxisTitle("Epoch")
            .yAxisTitle("Cross Entropy Loss")
            .build();

        chart.addSeries("Loss",epochs,losses);
        new SwingWrapper<>(chart).displayChart();
    }

    private static void plotAccuracy(
        double[] epochs,
        double[] trainAccuracy,
        double[] testAccuracy){

        XYChart chart=new XYChartBuilder()
            .width(900)
            .height(600)
            .title("MNIST Accuracy")
            .xAxisTitle("Epoch")
            .yAxisTitle("Accuracy")
            .build();

        chart.addSeries("Train Accuracy",epochs,trainAccuracy);
        chart.addSeries("Test Accuracy",epochs,testAccuracy);

        new SwingWrapper<>(chart).displayChart();
    }
}
