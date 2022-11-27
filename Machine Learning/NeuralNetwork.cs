using System;
using Newtonsoft.Json;

namespace Machine_Learning
{
    public class NeuralNetwork
    {
        private readonly int _inputNodeCount;
        private readonly int _hiddenNodeCount;
        private readonly int _outputNodeCount;
        private readonly double _learningRate;

        internal Matrix WeightsInputToHidden { get; set; }
        internal Matrix BiasHidden { get; set; }
        internal Matrix WeightsHiddenToOutput { get; set; }
        internal Matrix BiasOutput { get; set; }

        public NeuralNetwork(int inputs, int hidden, int outputs)
        {
            _inputNodeCount = inputs;
            _hiddenNodeCount = hidden;
            _outputNodeCount = outputs;
            _learningRate = 0.1;

            // Define Input->Hidden weights and bias.
            WeightsInputToHidden = new Matrix(_hiddenNodeCount, _inputNodeCount);
            BiasHidden = new Matrix(_hiddenNodeCount, 1);

            // Define Hidden->Output weights and bias.
            WeightsHiddenToOutput = new Matrix(_outputNodeCount, _hiddenNodeCount);
            BiasOutput = new Matrix(_outputNodeCount, 1);

            // Randomize matrices.            
            var rng = new Random();
            WeightsInputToHidden.Map(x => rng.NextDouble() * 2 - 1);
            BiasHidden.Map(x => rng.NextDouble() * 2 - 1);
            WeightsHiddenToOutput.Map(x => rng.NextDouble() * 2 - 1);
            BiasOutput.Map(x => rng.NextDouble() * 2 - 1);
        }

        private NeuralNetwork(int inputNodeCount, int hiddenNodeCount, int outputNodeCount, double learningRate, Matrix weightsInputToHidden, Matrix biasHidden, Matrix weightsHiddenToOutput, Matrix biasOutput)
        {
            _inputNodeCount = inputNodeCount;
            _hiddenNodeCount = hiddenNodeCount;
            _outputNodeCount = outputNodeCount;
            _learningRate = learningRate;
            WeightsInputToHidden = weightsInputToHidden;
            BiasHidden = biasHidden;
            WeightsHiddenToOutput = weightsHiddenToOutput;
            BiasOutput = biasOutput;
        }

        private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

        private double DSigmoid(double y) => y * (1 - y);

        public double[] FeedForward(double[] inputs)
        {
            var input = Matrix.FromArray(inputs);

            var hidden = Matrix.DotProduct(WeightsInputToHidden, input);
            hidden.Add(BiasHidden);
            hidden.Map(Sigmoid);

            var output = Matrix.DotProduct(WeightsHiddenToOutput, hidden);
            output.Add(BiasOutput);
            output.Map(Sigmoid);

            return output.ToArray();
        }

        public void Train(double[] inputs, double[] targets)
        {
            var input = Matrix.FromArray(inputs);

            var hidden = Matrix.DotProduct(WeightsInputToHidden, input);
            hidden.Add(BiasHidden);
            hidden.Map(Sigmoid);

            var output = Matrix.DotProduct(WeightsHiddenToOutput, hidden);
            output.Add(BiasOutput);
            output.Map(Sigmoid);

            var target = Matrix.FromArray(targets);
            var outputErrors = Matrix.Subtract(target, output);

            var outputGradient = Matrix.Map(output, DSigmoid);
            outputGradient.Multiply(outputErrors);
            outputGradient.Multiply(_learningRate);

            var hiddenOutputDeltas = Matrix.DotProduct(outputGradient, hidden.Transpose());
            WeightsHiddenToOutput.Add(hiddenOutputDeltas);
            BiasOutput.Add(outputGradient);

            var hiddenErrors = Matrix.DotProduct(WeightsHiddenToOutput.Transpose(), outputErrors);

            var hiddenGradient = Matrix.Map(hidden, DSigmoid);
            hiddenGradient.Multiply(hiddenErrors);
            hiddenGradient.Multiply(_learningRate);

            var inputHiddenDeltas = Matrix.DotProduct(hiddenGradient, input.Transpose());
            WeightsInputToHidden.Add(inputHiddenDeltas);
            BiasHidden.Add(hiddenGradient);
        }

        public string ToJson()
        {
            var output = JsonConvert.SerializeObject(new
            {
                InputNodeCount = _inputNodeCount,
                HiddenNodeCount = _hiddenNodeCount,
                OutputNodeCount = _outputNodeCount,
                LearningRate = _learningRate,
                WeightsInputToHidden = WeightsInputToHidden.Serialize(),
                BiasHidden = BiasHidden.Serialize(),
                WeightsHiddenToOutput = WeightsHiddenToOutput.Serialize(),
                BiasOutput = BiasOutput.Serialize()
            });

            return output;
        }

        public static NeuralNetwork FromJson(string data)
        {
            dynamic json = JsonConvert.DeserializeObject(data);

            var inputNodeCount = (int)json.InputNodeCount;
            var hiddenNodeCount = (int)json.HiddenNodeCount;
            var outputNodeCount = (int)json.OutputNodeCount;
            var learningRate = (double)json.LearningRate;
            var weightsInputToHidden = Matrix.Deserialize((string)json.WeightsInputToHidden);
            var biasHidden = Matrix.Deserialize((string)json.BiasHidden);
            var weightsHiddenToOutput = Matrix.Deserialize((string)json.WeightsHiddenToOutput);
            var biasOutput = Matrix.Deserialize((string)json.BiasOutput);

            return new NeuralNetwork(inputNodeCount, hiddenNodeCount, outputNodeCount, learningRate, weightsInputToHidden, biasHidden, weightsHiddenToOutput, biasOutput);
        }
    }
}