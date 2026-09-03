package core;
import exceptions.InvalidValue;

public class Matrix {
    private double[][] m;

    public Matrix(int n) throws InvalidValue {
        if (n <= 0) {
            throw new InvalidValue("Invalid Value");
        }
        this.m = new double[n][n];
    }

    public Matrix(Matrix m) {
        this.m = new double[m.shape()[0]][m.shape()[1]];

        for (int i = 0; i < m.shape()[0]; i++) {
            for (int j = 0; j < m.shape()[1]; j++) {
                this.m[i][j] = m.get(i, j);
            }
        }
    }

    public Matrix(double[] arr){
        this.m = new double[1][arr.length];
        int i = 0;
        for (double ele: arr){
            this.m[0][i] = ele;
            i++;
        } 
    }

    public Matrix(int m, int n) throws InvalidValue {
        if (n <= 0 || m <= 0) {
            throw new InvalidValue("Invalid Value");
        }
        this.m = new double[m][n]; 
    }

    public Matrix(double[][] arr) {
        this.m = new double[arr.length][arr[0].length];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                this.m[i][j] = arr[i][j];
            }
        }
    }

    public void ones() {
        for (int i = 0; i < this.m.length; i++) {
            for (int j = 0; j < this.m[0].length; j++) {
                this.m[i][j] = 1.0;
            }
        }
    }

    public static Matrix augment(Matrix m) throws InvalidValue{
        Matrix res = new Matrix(m.shape()[0], m.shape()[1] + 1);
        for(int i = 0; i < m.shape()[0]; i++){
            res.set(i,m.shape()[1], 1);
        }
        return res;
    }

    public Matrix flatten() throws InvalidValue {
        int totalElements = this.m.length * this.m[0].length;
        double[][] flatData = new double[1][totalElements];
        
        int index = 0;
        for (int i = 0; i < this.m.length; i++) {
            for (int j = 0; j < this.m[0].length; j++) {
                flatData[0][index++] = this.m[i][j];
            }
        }
        return new Matrix(flatData);
    }

    public static Matrix identity(int i) throws InvalidValue{
        Matrix I = new Matrix(i,i);
        for(int j = 0; j < i; j++){
            I.set(j, j, 1);
        }
        return I;
    }

    public String toString() {
        if (this.m.length != 1) {
            StringBuilder sb = new StringBuilder();
            sb.append('[');
            for (int i = 0; i < this.m.length; i++) {
                sb.append('[');
                for (int j = 0; j < this.m[i].length; j++) {
                    sb.append(this.m[i][j]);
                    if (j < this.m[i].length - 1) {
                        sb.append(", ");
                    }
                }
                sb.append(']');
                if (i < this.m.length - 1) {
                    sb.append(", ");
                }
            }
            sb.append(']');
            return sb.toString();
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append('[');
            for (int j = 0; j < this.m[0].length; j++) {
                sb.append(this.m[0][j]);
                if (j < this.m[0].length - 1) {
                    sb.append(", ");
                }
            }
            sb.append(']');
            return sb.toString();
        }
    }

    public int[] shape() {
        return new int[]{this.m.length, this.m[0].length};
    }

    public double[] getRow(int i) {
        return this.m[i];
    }

    public double[] getCol(int j) {
        double[] arr = new double[this.m.length];
        for (int i = 0; i < this.m.length; i++) {
            arr[i] = this.m[i][j];
        }
        return arr;
    }

    public void set(int i, int j, double val) {
        this.m[i][j] = val;
    }

    public void setRow(int i, double[] arr){
        for(int j = 0; j < this.getRow(i).length; j++){
            this.m[i][j] = arr[j];
        }
    }

    public double sumRow(int i){
        double[] arr = this.m[i];
        double res = 0.0;
        for(int j = 0; j < arr.length; j++)
            res += arr[j];
        return res;
    }

    public double get(int i, int j) {
        return this.m[i][j];
    }

    public Matrix getRows(int start, int end) throws InvalidValue{
        if(start < 0 || start >= this.m.length || end > this.m.length || end <= start){
            throw new InvalidValue("Invalid Indexing");
        }

        double[][] n = new double[end-start][this.shape()[1]];
        int index = 0;
        for(int i = start; i < end; i++){
            double[] arr = this.getRow(i);
            for(int j = 0; j < this.shape()[1]; j++){
                n[index][j] = arr[j];
            }
            index++;
        }
        return new Matrix(n);
    }

    public Matrix getRows(int[] indices) throws InvalidValue{
        if(indices==null||indices.length==0){
            throw new InvalidValue("Invalid Indices");
        }
        
        double[][] n=new double[indices.length][this.shape()[1]];
        for(int i=0;i<indices.length;i++){
            if(indices[i]<0||indices[i]>=this.m.length){
                throw new InvalidValue("Invalid Index");
            }
            double[] row=this.getRow(indices[i]);
            for(int j=0;j<this.shape()[1];j++){
                n[i][j]=row[j];
            }
        }
        return new Matrix(n);
    }

    public static double dot(double[] m1, double[] m2) throws InvalidValue {
        if (m1.length != m2.length) {
            throw new InvalidValue("Wrong Series");
        }
        double res = 0;
        for (int i = 0; i < m1.length; i++) {
            res += m1[i] * m2[i];
        }
        return res;
    }

    public static Matrix matmul(Matrix m1, Matrix m2) throws InvalidValue {
        int[] s1 = m1.shape();
        int[] s2 = m2.shape();

        if (s1[1] != s2[0]) {
            throw new InvalidValue("cannot perform matrix multiplication");
        }

        double[][] n = new double[s1[0]][s2[1]];
        for (int i = 0; i < s1[0]; i++) {
            for (int j = 0; j < s2[1]; j++) {
                n[i][j] = dot(m1.getRow(i), m2.getCol(j));
            }
        }
        return new Matrix(n);
    }

    private static boolean isValid(Matrix m1, Matrix m2){
        for (int i = 0; i < m1.shape().length; i++){
            if (m1.shape()[i] != m2.shape()[i])
                return false;
        }
        return true;
    }

    public static Matrix Transpose(Matrix ma) throws InvalidValue{
        Matrix n = new Matrix(ma.shape()[1], ma.shape()[0]);
        for(int i =0; i < ma.shape()[0]; i++){
            for (int j = 0; j < ma.shape()[1]; j++){
                n.set(j, i, ma.get(i,j));
            }
        }
        return n;
    }

    public static Matrix add(Matrix m1, Matrix m2) throws InvalidValue{
        if (!isValid(m1, m2))
            throw new InvalidValue("Wrong Matrices");
        Matrix m = new Matrix(m1.shape()[0], m2.shape()[1]);
        for( int i =0; i < m1.shape()[0]; i++){
            for(int j = 0; j < m2.shape()[1]; j++){
                m.set(i,j, m1.get(i,j) + m2.get(i, j));
            }
        }
        return m;
    }

    public static Matrix sub(Matrix m1, Matrix m2) throws InvalidValue{
        if (!isValid(m1, m2))
            throw new InvalidValue("Wrong Matrices");
        Matrix m = new Matrix(m1.shape()[0], m2.shape()[1]);
        for( int i =0; i < m1.shape()[0]; i++){
            for(int j = 0; j < m2.shape()[1]; j++){
                m.set(i,j, m1.get(i,j) - m2.get(i, j));
            }
        }
        return m;
    }

    public static Matrix ones(int i, int j) throws InvalidValue{
        Matrix res = new Matrix(i,j);
        for(int k = 0; k < i; k++){
            for(int l = 0; l < j; l++){
                res.set(k, l, 1);
            }
        }
        return res;
    }

    public static double[][] scalerMul(Matrix m, double n){
        double[][] ma = new double[m.shape()[0]][m.shape()[1]];
        for(int i = 0; i < m.shape()[0]; i++){
            for(int j = 0; j < m.shape()[1]; j++){
                ma[i][j] = m.get(i, j) *  n;
            }
        }
        return ma;
    }

    public static Matrix Hadamard(Matrix m1, Matrix m2) throws InvalidValue{
        if (!isValid(m1, m2))
            throw new InvalidValue("Wrong Matrices");
        Matrix m = new Matrix(m1.shape()[0], m1.shape()[1]);
        for(int i = 0; i < m.shape()[0]; i++){
            for(int j = 0; j < m.shape()[1]; j++){
                m.set(i, j, m1.get(i,j) * m2.get(i, j));
            }
        }
        return m;
    }

    public static void main(String[] args) {
        try {
            Matrix m = new Matrix(2,3);
            System.out.println(augment(m));
        } catch (InvalidValue e) {
            System.out.println(e.getMessage());
        }
    }
}
