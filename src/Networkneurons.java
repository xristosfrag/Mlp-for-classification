import java.util.ArrayList;
import java.util.Scanner;
import java.util.Random;
import java.lang.Math;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.FileInputStream;

public class Networkneurons {

	private ArrayList<double[]> allLevelNeurons;
	private ArrayList<double[]> bias;
	private ArrayList<double[]> allLevelErrors;
	private ArrayList<double[][]> allLevelWeights;
	private ArrayList<double[]> allDerivatives;
	private static int hidden_layers = 0;
	private int total_layers;
	private static String activationFunction = "";
	private static int categories = 0;
	private int current_category;
	private double[][] trainingPoints;
	private double[][] testPoints;
	private int points;
	private double learningRate = 0.1;
	private int succeed = 0;
	private double MSE = 1;

	public Networkneurons(int points) {
		this.points = points;
		total_layers = Networkneurons.hidden_layers + 2;
		allLevelNeurons = initializeNeurons(0);
		allLevelErrors = initializeNeurons(1);
		allLevelWeights = initializeWeights();
		bias = initializebias();
		produceTrainingPoints();
		produceTestPoints();
		trainingPoints = loadTrainingPoints();
		testPoints = loadTrainingPoints();
		allDerivatives = initializeNeurons(1);
	}

	public static void main(String[] args) {

		scanInput();
		Networkneurons mlp = new Networkneurons(4000);
		System.out.println(" PLEAS WAIT, WHILE MLP GETS TRAINED ...");
		mlp.trainNeuralNetwork();
		mlp.testNeuralNetwork();

	}

	// SCAN INPUT PARAMETERS FOR MLP
	// ====================================================================================================

	public static void scanInput() {
		Scanner in = new Scanner(System.in);
		System.out.println(
				" Welcome to the multilayer perceptron! Please insert your activation function (tanh or sigmoid)");

		Networkneurons.activationFunction = in.nextLine();

		int flag = 0;
		while (flag == 0) {
			if (Networkneurons.activationFunction.equals("tanh")
					|| Networkneurons.activationFunction.equals("sigmoid")) {
				flag = 1;
			} else {
				System.out.println(
						" Input  function incorrect! Please insert again your activation function (tanh or sigmoid)");
				Networkneurons.activationFunction = in.nextLine();
			}
		}
		flag = 0;
		System.out.println(" Choose the number of hidden layers! Only positive numbers:");
		Networkneurons.hidden_layers = in.nextInt();

		while (flag == 0) {
			if (Networkneurons.hidden_layers > 0) {
				flag = 1;
			} else {
				System.out.println(" Input  number incorrect! Please insert again");
				Networkneurons.hidden_layers = in.nextInt();
			}
		}
		flag = 0;
		System.out.println(" Choose the number of output categories! Only positive numbers:");
		Networkneurons.categories = in.nextInt();
		while (flag == 0) {
			if (Networkneurons.categories > 0) {
				flag = 1;
			} else {
				System.out.println(" Input  number incorrect! Please insert again");
				Networkneurons.categories = in.nextInt();
			}
		}
		in.close();
	}

	// ====================================================================================================

	// FORWARDPASS FUNCTION
	// ====================================================================================================

	private void forwardPass(int EpochIterator, int trainOrTest) {

		if (trainOrTest == 0) {
			allLevelNeurons.get(0)[0] = trainingPoints[EpochIterator][0];
			allLevelNeurons.get(0)[1] = trainingPoints[EpochIterator][1];
		} else {
			allLevelNeurons.get(0)[0] = testPoints[EpochIterator][0];
			allLevelNeurons.get(0)[1] = testPoints[EpochIterator][1];

		}

		// FINDING THE EXPECTED CATEGORY
		findCategory(trainingPoints[EpochIterator][0], trainingPoints[EpochIterator][1]);

		for (int i = 1; i < allLevelNeurons.size(); i++) {
			for (int j = 0; j < allLevelNeurons.get(i).length; j++) {
				double sum = bias.get(i)[j];
				for (int k = 0; k < allLevelNeurons.get(i - 1).length; k++) {
					sum += allLevelNeurons.get(i - 1)[k] * allLevelWeights.get(i)[j][k];

				}

				if (Networkneurons.activationFunction.equals("tanh")) {

					allLevelNeurons.get(i)[j] = Math.tanh(sum);

					// derivative of the tanh function
					allDerivatives.get(i)[j] = 1 - (allLevelNeurons.get(i)[j] * allLevelNeurons.get(i)[j]);
				} else {
					allLevelNeurons.get(i)[j] = sigmoid(sum);

					// derivative of the sigmoid function
					allDerivatives.get(i)[j] = allLevelNeurons.get(i)[j] * (1 - allLevelNeurons.get(i)[j]);
				}

			}

		}

		// print OutPutNeurons
		// if = 0 only needed if programmer wants to check/DEBUG training iterrations
		if (trainOrTest == 0) {
			// if(EpochIterator == 99) {
			// for(int i=0; i<allLevelNeurons.get(allLevelNeurons.size() -1 ).length; i++)
			// {
			// int g = i + 1;
			// System.out.println(" neuron " + g + " ----> "
			// +allLevelNeurons.get(allLevelNeurons.size() -1 )[i]);
			// System.out.println("catecory " + current_category);
			// }
			// }
		} else {
			System.out.println("catecory " + current_category);
			for (int i = 0; i < allLevelNeurons.get(allLevelNeurons.size() - 1).length; i++) {
				int g = i + 1;
				System.out.println(" neuron  " + g + "  ---->  " + allLevelNeurons.get(allLevelNeurons.size() - 1)[i]);

			}

			if (current_category == 1) {
				if (allLevelNeurons.get(total_layers - 1)[current_category - 1] > 0.85) {
					succeed++;
				}
			} else if (current_category == 2) {
				if (allLevelNeurons.get(total_layers - 1)[current_category - 1] > 0.85) {
					succeed++;
				}
			} else if (current_category == 3) {
				if (allLevelNeurons.get(total_layers - 1)[current_category - 1] > 0.85) {
					succeed++;
				}
			} else {
				if (allLevelNeurons.get(total_layers - 1)[current_category - 1] > 0.85) {
					succeed++;
				}
			}

		}

	}
	// ====================================================================================================

	// BACKPROPAGATION FUNCTION
	// ====================================================================================================

	private void backPropagation(int x) {
		forwardPass(x, 0);
		MSE = 0;
		for (int i = 0; i < allLevelNeurons.get(total_layers - 1).length; i++) {
			if ((i + 1) == current_category) {
				allLevelErrors.get(total_layers - 1)[i] = (allLevelNeurons.get(total_layers - 1)[i] - 1)
						* allDerivatives.get(total_layers - 1)[i];
				MSE += (allLevelNeurons.get(total_layers - 1)[i] - 1) * (allLevelNeurons.get(total_layers - 1)[i] - 1);
			} else {
				allLevelErrors.get(total_layers - 1)[i] = (allLevelNeurons.get(total_layers - 1)[i] - 0)
						* allDerivatives.get(total_layers - 1)[i];
				MSE += (allLevelNeurons.get(total_layers - 1)[i] - 0) * (allLevelNeurons.get(total_layers - 1)[i] - 0);
			}

		}
		for (int j = total_layers - 2; j > 0; j--) {
			for (int k = 0; k < allLevelNeurons.get(j).length; k++) {
				double sum = 0;
				for (int m = 0; m < allLevelNeurons.get(j + 1).length; m++) {
					sum += allLevelWeights.get(j + 1)[m][k] * allLevelErrors.get(j + 1)[m];
				}
				allLevelErrors.get(j)[k] = sum * allDerivatives.get(j)[k];
			}
		}

		for (int z = 1; z < total_layers; z++) {
			for (int y = 0; y < allLevelNeurons.get(z).length; y++) {
				for (int f = 0; f < allLevelNeurons.get(z - 1).length; f++) {
					double deltaWeight = -learningRate * allLevelNeurons.get(z - 1)[f] * allLevelErrors.get(z)[y];
					allLevelWeights.get(z)[y][f] += deltaWeight;
				}
				double deltaBias = -learningRate * allLevelErrors.get(z)[y];
				bias.get(z)[y] += deltaBias;
			}
		}
	}

	// ====================================================================================================

	// TRAIN THE NETWORK
	// ====================================================================================================

	private void trainNeuralNetwork() {

		for (int epoch = 0; epoch < 1000; epoch++)

		{
			for (int i = 0; i < 4000; i++) {

				backPropagation(i);
				System.out.println("MSE =  " + MSE);
			}

		}

	}
	// ====================================================================================================

	// TEST THE NETWORK
	// ====================================================================================================
	private void testNeuralNetwork() {

		for (int i = 0; i < points; i++) {

			forwardPass(i, 1);
		}
		System.out.println(succeed);
		float t = succeed / 40;
		System.out.println("success percentage of MLP = " + t + " % ");

	}

	// ====================================================================================================

	// SIGMOID FUNCTION
	private double sigmoid(double x) {
		return (1d / (1 + Math.exp(-x)));
	}

	// FIND CATEGORY OF CURRENT POINT
	// ================================================================
	private void findCategory(double x1, double x2) {

		if ((Math.pow((x1 - 0.5), 2) + Math.pow((x2 - 0.5), 2)) < 0.16) {

			current_category = 1; // CATEGORY 1

		} else if ((Math.pow((x1 + 0.5), 2) + Math.pow((x2 + 0.5), 2)) < 0.16) {
			current_category = 1; // CATEGORY 1
		} else if ((Math.pow((x1 - 0.5), 2) + Math.pow((x2 + 0.5), 2)) < 0.16) {
			current_category = 2; // CATEGORY 2
		} else if ((Math.pow((x1 + 0.5), 2) + Math.pow((x2 - 0.5), 2)) < 0.16) {
			current_category = 2; // CATEGORY 2
		} else if (((x1 >= 0) && (x2 >= 0)) || ((x1 <= 0) && (x2 <= 0))) {
			current_category = 3; // CATEGORY 3
		} else {
			current_category = 4; // CATEGORY 4
		}

	}

	// =============================================================================================

	// POINTS FACTORY (BOTH TRAINING AND TESTING)
	// ====================================================================================================

	protected static void produceTrainingPoints() {
		FileOutputStream outputStream = null;
		try {
			outputStream = new FileOutputStream("trainingPoints.txt");
		} catch (FileNotFoundException e) {
			System.out.println("Error in opening the file points.txt");
			System.exit(0);
		}
		PrintWriter outputWriter = new PrintWriter(outputStream);

		// System.out.println("Writing to file.");
		Random random = new Random();
		for (int i = 0; i < 4000; i++) {

			double x1 = -1 + random.nextDouble() * (2);
			double x2 = -1 + random.nextDouble() * (2);

			outputWriter.println(x1 + "," + x2);
			outputWriter.flush();
		}
	}

	protected static void produceTestPoints() {
		FileOutputStream outputStream = null;
		try {
			outputStream = new FileOutputStream("testPoints.txt");
		} catch (FileNotFoundException e) {
			System.out.println("Error in opening the file points.txt");
			System.exit(0);
		}
		PrintWriter outputWriter = new PrintWriter(outputStream);

		// System.out.println("Writing to file.");
		Random random = new Random();
		for (int i = 0; i < 4000; i++) {

			double x1 = -1 + random.nextDouble() * (2);
			double x2 = -1 + random.nextDouble() * (2);

			outputWriter.println(x1 + "," + x2);
			outputWriter.flush();
		}
	}

	protected static double[][] loadTrainingPoints() {
		Scanner inputStream = null;
		try {
			inputStream = new Scanner(new FileInputStream("trainingPoints.txt"));
		} catch (FileNotFoundException e) {
			System.out.println(
					"File " + "trainingPoints.txt" + " was not found\nor could not be opened\nSystem Exits...");
			System.exit(0);
		}
		String line = "";

		double[][] trainingPoints = new double[4000][2];

		System.out.println("Reading from file");
		int i = 0;
		while (inputStream.hasNextLine() && i < 4000) {
			int j = 0;
			while (j < 1) {
				line = inputStream.nextLine();
				String[] point = line.split(",");
				double x1 = Double.parseDouble(point[0]);
				double x2 = Double.parseDouble(point[1]);
				trainingPoints[i][j] = x1;
				trainingPoints[i][j + 1] = x2;
				j++;
			}
			i++;
		}
		return trainingPoints;
	}

	protected static double[][] loadTestPoints() {
		Scanner inputStream = null;
		try {
			inputStream = new Scanner(new FileInputStream("testPoints.txt"));
		} catch (FileNotFoundException e) {
			System.out.println("File " + "testPoints.txt" + " was not found\nor could not be opened\nSystem Exits...");
			System.exit(0);
		}
		String line = "";

		double[][] testPoints = new double[4000][2];

		System.out.println("Reading from file");
		int i = 0;
		while (inputStream.hasNextLine() && i < 4000) {
			int j = 0;
			while (j < 1) {
				line = inputStream.nextLine();
				String[] point = line.split(",");
				double x1 = Double.parseDouble(point[0]);
				double x2 = Double.parseDouble(point[1]);
				testPoints[i][j] = x1;
				testPoints[i][j + 1] = x2;
				j++;
			}
			i++;
		}
		return testPoints;
	}

	// ====================================================================================================

	// INITIALIZE BIAS
	// ====================================================================================================
	private ArrayList<double[]> initializebias() {
		Random random = new Random();
		ArrayList<double[]> arrayListBiases = new ArrayList<double[]>();

		for (int i = 0; i < total_layers; i++) {
			if (i == 0) {
				double[] biasArray = new double[2];
				arrayListBiases.add(biasArray);
			} else {
				double[] biasArray = new double[allLevelNeurons.get(i).length];
				for (int j = 0; j < biasArray.length; j++) {
					biasArray[j] = -1 + random.nextDouble() * (1 - (-1));
				}
				arrayListBiases.add(biasArray);
			}
		}
		return arrayListBiases;
	}
	// ====================================================================================================

	// INITIALIZE NEURONS
	// ====================================================================================================
	private ArrayList<double[]> initializeNeurons(int x) {
		ArrayList<double[]> arrayListNeurons = new ArrayList<double[]>();
		Scanner cat = new Scanner(System.in);

		for (int i = 0; i < total_layers; i++) {
			if (i == 0) {
				double[] neuronArray = new double[2];
				arrayListNeurons.add(neuronArray);
			} else if (i == (total_layers - 1)) {
				double[] neuronArray = new double[categories];
				arrayListNeurons.add(neuronArray);
			} else {

				// errorArray or NeuronArray
				if (x == 0) {
					System.out.println("Enter the number of neurons in hidden level   " + i + ": ");
					int num = cat.nextInt();
					double[] neuron = new double[num];
					arrayListNeurons.add(neuron);
				} else if (x == 1) {
					double[] neuron = new double[allLevelNeurons.get(i).length];
					arrayListNeurons.add(neuron);
				}

			}

		}
		cat.close();
		return arrayListNeurons;
	}
	// ====================================================================================================

	// INITIALIZE WEIGHTS
	// ==============================================================================================
	private ArrayList<double[][]> initializeWeights() {
		ArrayList<double[][]> arrayListWeight = new ArrayList<double[][]>();
		for (int i = 0; i < total_layers; i++) {
			if (i == 0) {
				double[][] weight0 = new double[0][0];
				arrayListWeight.add(weight0);
			} else {
				double[][] weight = new double[allLevelNeurons.get(i).length][allLevelNeurons.get(i - 1).length];
				weight = randomizeWeightsValues(weight, allLevelNeurons.get(i).length,
						allLevelNeurons.get(i - 1).length);
				arrayListWeight.add(weight);
			}
		}
		return arrayListWeight;
	}

	private double[][] randomizeWeightsValues(double[][] x, int rows, int columns) {
		Random random = new Random();
		for (int z = 0; z < rows; z++) {
			for (int y = 0; y < columns; y++) {
				x[z][y] = -1 + random.nextDouble() * (1 - (-1));

			}

		}
		return x;
	}
}