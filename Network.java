import java.util.Random;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Scanner;
public class Network {
	double[][] trainingData = {
			{0,1,0},
			{1,0,0},
			{0,0,0},
			{1,1,1}
		};
	
	Random rand = new Random();
	double[][][] weights = new double[2][][];
	double[] hiddenLayer = new double[785];
	double[] inputLayer = new double[785];
	double[] expectedOutput = new double[10];
	double[] outputLayer = new double[10];
	double learningRate = 0.65;
	double[] deltaHidden;
	double[][] trainingDataMNIST = new double[60000][785];
	
	public Network(){
		try {
			Scanner s = new Scanner(new File("mnist_train.txt")).useDelimiter("\\n");
			for(int r = 0; r<60000; r++){
				String[] line = s.next().split(",");
				for (int i=0; i < line.length; i++) {
			        trainingDataMNIST[r][i] = Double.parseDouble(line[i]);
			    }
			}
		} catch(FileNotFoundException e){}
		
		// input layer neurons loop
		weights[0] = new double[inputLayer.length][hiddenLayer.length];
		for(int a = 0; a < inputLayer.length; a++){
			// input layer weights loop
			for(int b = 0; b < hiddenLayer.length; b++){
				weights[0][a][b] = rand.nextInt(10);
			}
		}
		
		// hidden layer neurons loop
		weights[1] = new double[hiddenLayer.length][10];
		for(int i = 0; i < hiddenLayer.length; i++){
			for(int j = 0; j < outputLayer.length; j++){
				// hidden layer weights loop
				weights[1][i][j] = rand.nextInt(10);
			}
		}
		long start = System.currentTimeMillis();
		for(int c = 0; c<600; c++){
			int index = -1;
			for(double[] t:trainingDataMNIST){
				index++;
				expectedOutput = new double[10];
				if(t[0] != 0){
					expectedOutput[((int)t[0])-1] = 1;
				}else{
					expectedOutput[9] = 1;
				}
				for(int i = 1; i<inputLayer.length; i++){
					inputLayer[i] = t[i];
				}
				
				hiddenLayer = new double[785];
				
				for(int k = 0; k < inputLayer.length; k++){
					for(int l = 0; l < hiddenLayer.length; l++){
						hiddenLayer[l] += weights[0][k][l] * inputLayer[k];
					}
				}
				// apply activation function on all sums and check if they pass the threshold of 0
				for(int m = 0; m < hiddenLayer.length; m++){
					hiddenLayer[m] = sig(hiddenLayer[m]);
				}

				// repeat with the hidden layer to output layer
				double[] predictedOutput = new double[10];
				for(int k = 0; k < hiddenLayer.length; k++){
					for(int l = 0; l < predictedOutput.length; l++){
						predictedOutput[l] += weights[1][k][l] * hiddenLayer[k];
					}
				}

				double[] output = new double[10];
				// apply activation function on all sums and check if they pass the threshold of 0
				for(int a = 0; a < predictedOutput.length; a++){
					output[a] = sig(predictedOutput[a]);
				}
				// error
				System.out.println("Index: " + index);
				double[] deltaOutput = new double[10];
				for(int i = 0; i < deltaOutput.length; i++){
					deltaOutput[i] = output[i] - expectedOutput[i];
					System.out.println(deltaOutput[i]);
				}
				System.out.println("—————————————");
				
				//System.out.println(output);
				//System.out.println(expectedOutput);
				//System.out.println();
				
				double[] deltaHidden = new double[hiddenLayer.length];

				// Backprop
				for(int i = 0; i < hiddenLayer.length; i++){
					double delta = 0;
					for(int j = 0; j < outputLayer.length; j++){
						delta += deltaOutput[j] * weights[1][i][j] * hiddenLayer[i] * (1-hiddenLayer[i]);
					}
					deltaHidden[i] = delta;
				}
				for(int i = 0; i < weights[1].length; i++){
					for(int j = 0; j < deltaOutput.length; j++){
						weights[1][i][j] -= learningRate * hiddenLayer[i] * deltaOutput[j];
					}
				}
				
				for(int i = 0; i < weights[0].length; i++){
					for(int j = 0; j < deltaHidden.length; j++){
						weights[0][i][j] -= learningRate * inputLayer[j] * deltaHidden[j];
					}
				}

			}
		}
		System.out.println("Time: " + (System.currentTimeMillis() - start));
	}
	public double[] predict(int[] u){
		inputLayer = new double[784];
		for(int i = 1; i<inputLayer.length; i++){
			inputLayer[i] = u[i];
		}
		
		hiddenLayer = new double[785];
		
		for(int k = 0; k < inputLayer.length; k++){
			for(int l = 0; l < hiddenLayer.length; l++){
				hiddenLayer[l] += weights[0][k][l] * inputLayer[k];
			}
		}
		// apply activation function on all sums and check if they pass the threshold of 0
		for(int m = 0; m < hiddenLayer.length; m++){
			hiddenLayer[m] = sig(hiddenLayer[m]);
		}

		// repeat with the hidden layer to output layer
		double[] predictedOutput = new double[10];
		for(int k = 0; k < hiddenLayer.length; k++){
			for(int l = 0; l < predictedOutput.length; l++){
				predictedOutput[l] += weights[1][k][l] * hiddenLayer[k];
			}
		}

		double[] output = new double[10];
		// apply activation function on all sums and check if they pass the threshold of 0
		for(int a = 0; a < predictedOutput.length; a++){
			output[a] = sig(predictedOutput[a]);
		}
		return output;
	}
//	public double test(int[][] testData){
//		int correct = 0;
//		int total = testData.length;
//		for(int i = 0; i < testData.length; i++){
//			int[] inputs = {testData[i][0],testData[i][1]};
//			int expected = testData[i][2];
//			int predicted = predict(inputs);
//			if(expected == predicted){
//				correct++;
//			}
//		}
//		double accuracy = (double) correct/total;
//		return accuracy;
//	}
	public static void main(String[] args){
		Network net = new Network();
//		int[][] inputs = {
//				{0,1,1},
//				{1,0,1},
//				{0,0,0},
//				{1,1,1}
//				};
		int[] inputs = {5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		double[] prediction = net.predict(inputs);
		System.out.println();
		FileWriter fileWriter;
		try {
			fileWriter = new FileWriter("weights.txt");
			PrintWriter printWriter = new PrintWriter(fileWriter);
			for(int i = 0; i < net.weights.length; i++){
				for(int j = 0; j < net.weights[i].length; j++){
					for(int k = 0; k < net.weights[i][j].length; k++){
						printWriter.println(i + "," + j + "," + k);
					}
				}
			}
			printWriter.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for(int i = 0; i < prediction.length; i++){
			System.out.print(prediction[i] + " ");
		}
	}
	private double sig(double sum){
		return 1/(1+Math.exp(-sum));
	}
	public void printNet(){
		System.out.println("Initializing Forward Propagation...");
		System.out.print("Input Layer: ");
		for(double i:inputLayer){
			System.out.print(i + " ");
		}
		System.out.println();
		System.out.println();
		
		System.out.print("First Neuron Weights: ");
		
		for(double j:weights[0][0]){
			System.out.print(j + " ");
		}
		
		System.out.println();
		System.out.print("Second Neuron Weights: ");
		
		for(double k:weights[0][1]){
			System.out.print(k + " ");
		}
		
		System.out.println();
		System.out.println();
		
		System.out.print("Hidden Layer: ");
		for(double l:hiddenLayer){
			System.out.print(l + " ");
		}
		System.out.println();
		System.out.println();
		
		System.out.print("Hidden Layer First Neuron Weights: ");
		for(double l:weights[1][0]){
			System.out.print(l + " ");
		}
		System.out.println();
		
		System.out.print("Hidden Layer Second Neuron Weights: ");
		for(double l:weights[1][1]){
			System.out.print(l + " ");
		}
		
		System.out.println();
		System.out.print("Hidden Layer Third Neuron Weights: ");
		for(double l:weights[1][2]){
			System.out.print(l + " ");
		}
		
		System.out.println();
		System.out.println();
		
		//System.out.println("Output: " + output);
		
		System.out.println("Initializing Backpropagation...");
		
		
		
		System.out.println("——————————————————————————————————————————————————————————");
	}
}
