using System;

namespace Machine_Learning
{
    public class Evolution
    {
        private static double _standardDeviation = 0.001;

        public static void SetStandardDeviation(double standardDeviation)
        {
            _standardDeviation = standardDeviation;
        }

        public static NeuralNetwork Crossover(NeuralNetwork networkA, NeuralNetwork networkB)
        {
            throw new NotImplementedException();
        }

        public static void Mutate(NeuralNetwork neuralNetwork, double mutationRate)
        {
            if (mutationRate < 0 || mutationRate > 1) throw new ArgumentOutOfRangeException();

            var rng = new Random();

            // Mutate weights.
            neuralNetwork.WeightsInputToHidden.Map(value => rng.NextDouble() <= mutationRate ? value + GaussianRandom(0, _standardDeviation) : value);
            neuralNetwork.WeightsHiddenToOutput.Map(value => rng.NextDouble() <= mutationRate ? value + GaussianRandom(0, _standardDeviation) : value);

            // Mutate bias.
            neuralNetwork.BiasHidden.Map(value => rng.NextDouble() <= mutationRate ? value + GaussianRandom(0, _standardDeviation) : value);
            neuralNetwork.BiasOutput.Map(value => rng.NextDouble() <= mutationRate ? value + GaussianRandom(0, _standardDeviation) : value);
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