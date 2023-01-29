using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Machine_Learning
{
    public class NeuralNetwork
    {
        private readonly int _inputs;
        private readonly int _outputs;
        public readonly List<Node> Nodes = new List<Node>();
        private static readonly Random rng = new Random();
        private static double _standardDeviation = 1.0;

        public NeuralNetwork(int inputs, int outputs)
        {
            _inputs = inputs;
            _outputs = outputs;

            // Initialize input and output nodes.
            for (int i = 0; i < _inputs + _outputs; i++)
            {
                AddNode();
            }
        }

        public void Initialize()
        {
            // Wire inputs to outputs.
            for (var j = 0; j < _inputs; j++)
            {
                for (var k = _inputs; k < _inputs + _outputs; k++)
                {
                    AddConnection(j, k);
                }
            }
        }

        public void AddNode()
        {
            Nodes.Add(new Node(Nodes.Count));
        }

        public void AddConnection(int fromNodeId, int toNodeId)
        {
            var weight = rng.NextDouble() * 2.0 - 1.0;

            AddConnection(fromNodeId, toNodeId, weight);
        }

        public void AddConnection(int fromNodeId, int toNodeId, double weight)
        {
            var fromNode = Nodes[fromNodeId];
            var toNode = Nodes[toNodeId];

            toNode.Connections.Add(new Connection(fromNode, toNode, weight));
        }

        public double[] Evaluate(double[] inputs)
        {
            if (inputs.Length != _inputs) throw new ArgumentOutOfRangeException();

            // Set input node values.
            for (var i = 0; i < _inputs; i++)
            {
                Nodes[i].SetValue(inputs[i]);
            }

            // Calculate output node values.
            var output = new double[_outputs];

            for (var i = 0; i < _outputs; i++)
            {
                var node = Nodes[_inputs + i];
                var value = node.CalculateValue();
                output[i] = value;
            }

            // Get sum of outputs
            var sum = output.Sum();

            var results = new double[output.Length];

            for (var i = 0; i < output.Length; i++) results[i] = output[i] / sum;

            return output;
        }

        public NeuralNetwork Clone()
        {
            var clone = new NeuralNetwork(_inputs, _outputs);

            // Add any hidden nodes.
            for (var i = _inputs + _outputs; i < Nodes.Count; i++)
            {
                clone.AddNode();
            }

            // Find all connections between nodes.
            foreach (var node in Nodes)
            {
                foreach (var connection in node.Connections)
                {
                    // Find indexes for the from and to nodes.
                    var fromNode = clone.Nodes[Nodes.IndexOf(connection.FromNode)];
                    var toNode = clone.Nodes[Nodes.IndexOf(connection.ToNode)];

                    toNode.Connections.Add(new Connection(fromNode, toNode, connection.Weight));
                }
            }

            return clone;
        }

        private IEnumerable<Connection> GetConnections()
        {
            return Nodes.SelectMany(node => node.Connections);
        }

        public void MutateNodes(double mutationRate)
        {
            if (!(rng.NextDouble() < mutationRate)) return;

            // Get all current connection.
            var connections = GetConnections().ToList();

            // Find a connection to split.
            var targetConnection = connections[rng.Next(0, connections.Count)];

            // Remove connection from end node.
            var endNode = targetConnection.ToNode;
            endNode.Connections.Remove(targetConnection);

            // Add a new node.
            AddNode();
            var newNode = Nodes[Nodes.Count - 1];

            // Create connection from original start node.
            var startNode = targetConnection.FromNode;
            newNode.Connections.Add(new Connection(startNode, newNode, targetConnection.Weight));

            // Create connection from new node to original end node.
            endNode.Connections.Add(new Connection(newNode, endNode, rng.NextDouble() * 2.0 - 1.0));
        }

        public void MutateConnections(double mutationRate)
        {
            if (!(rng.NextDouble() < mutationRate)) return;

            // Find a node to act as the end node.
            var endNode = Nodes[rng.Next(_inputs, Nodes.Count)];
            var startNode = endNode;

            // Find a node that isn't the end node.
            while (startNode == endNode)
            {
                startNode = Nodes[rng.Next(0, Nodes.Count)];
            }

            // Check if a connection already exists
            var connectionId = $"{startNode.Id}->{endNode.Id}";
            var connectionExists = endNode.Connections.Any(connection => connection.Id == connectionId);

            if (!connectionExists)
            {
                // Make sure we're not making an infinite loop.
                var nodes = new Queue<Node>();
                nodes.Enqueue(startNode);
                var foundIt = false;

                while (nodes.Count > 0 && !foundIt)
                {
                    var node = nodes.Dequeue();

                    foreach (var connection in node.Connections)
                    {
                        if (connection.FromNode != endNode) continue;

                        foundIt = true;
                        break;
                    }

                    // Stop if we found the end node "up stream" from the start.
                    if (foundIt) continue;

                    foreach (var connection in node.Connections)
                    {
                        nodes.Enqueue(connection.FromNode);
                    }
                }

                if (!foundIt)
                {
                    // Create a connection.
                    var weight = rng.NextDouble() * 2.0 - 1.0;
                    endNode.Connections.Add(new Connection(startNode, endNode, weight));
                }
            }
        }

        public void MutateWeights(double mutationRate)
        {
            foreach (var connection in Nodes.SelectMany(item => item.Connections))
            {
                if (!(rng.NextDouble() < mutationRate)) return;

                if (rng.NextDouble() < 0.2)
                {
                    // Randomize weight.
                    connection.Weight = rng.NextDouble() * 2.0 - 1.0;
                }
                else
                {
                    // Adjust weight.
                    connection.Weight += GaussianRandom(0, _standardDeviation);
                }
            }
        }

        private static double GaussianRandom(double mean, double stdDev)
        {
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }

        public string Serialize()
        {
            var output = "";

            // Collect network meta data.
            output += $"{_inputs},{_outputs},{Nodes.Count - _inputs - _outputs}";

            // Build connection list.
            var connections = Nodes.SelectMany(node => node.Connections).ToList();

            foreach (var connection in connections)
            {
                output += $",{connection.Weight}|{connection.FromNode.Id}|{connection.ToNode.Id}";
            }

            return output;
        }

        public static NeuralNetwork Deserialize(string input)
        {
            var pieces = input.Split(',');

            var inputCount = int.Parse(pieces[0]);
            var outputCount = int.Parse(pieces[1]);
            var totalHidden = int.Parse(pieces[2]);

            var nn = new NeuralNetwork(inputCount, outputCount);

            // Create hidden nodes.
            for (var i = 0; i < totalHidden; i++)
            {
                nn.AddNode();
            }

            // Build connections.
            var connections = pieces.Skip(3).ToList();

            foreach (var connectionInfo in connections)
            {
                var connection = connectionInfo.Split('|');
                var weight = double.Parse(connection[0]);
                var fromId = int.Parse(connection[1]);
                var toId = int.Parse(connection[2]);

                nn.AddConnection(fromId, toId, weight);
            }

            return nn;
        }

        public static void SetStandardDeviation(double standardDeviation)
        {
            _standardDeviation = standardDeviation;
        }
    }
}