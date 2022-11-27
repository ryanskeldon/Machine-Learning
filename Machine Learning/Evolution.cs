using System;

namespace Machine_Learning
{
    public class Evolution
    {
        public static NeuralNetwork Crossover(NeuralNetwork networkA, NeuralNetwork networkB)
        {
            throw new NotImplementedException();
        }

        public static void Mutate(NeuralNetwork neuralNetwork, double mutationRate)
        {
            if (mutationRate < 0 || mutationRate > 1) throw new ArgumentOutOfRangeException();

            // Mutate weights.
            neuralNetwork.WeightsInputToHidden.Map((value) => new Random().NextDouble() <= mutationRate ? GaussianRandom(0, 0.001) : value);
            neuralNetwork.WeightsHiddenToOutput.Map((value) => new Random().NextDouble() <= mutationRate ? GaussianRandom(0, 0.001) : value);

            // Mutate bias.
            neuralNetwork.BiasHidden.Map((value) => new Random().NextDouble() <= mutationRate ? GaussianRandom(0, 0.001) : value);
            neuralNetwork.BiasOutput.Map((value) => new Random().NextDouble() <= mutationRate ? GaussianRandom(0, 0.001) : value);
        }

        private static double GaussianRandom(double mean, double stdDev)
        {
            var rand = new Random();
            var u1 = 1.0 - rand.NextDouble();
            var u2 = 1.0 - rand.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}