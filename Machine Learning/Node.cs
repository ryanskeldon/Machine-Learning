using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Machine_Learning
{
    public class Node
    {
        public int Id { get; }
        public List<Connection> Connections { get; set; }
        private double? _output;

        public Node(int id)
        {
            Id = id;
            Connections = new List<Connection>();
        }

        public double CalculateValue()
        {
            if (_output != null) return _output.Value;

            var sum = 0.0;

            foreach (var connection in Connections)
            {
                sum += connection.Weight * connection.FromNode.CalculateValue();
            }

            return Sigmoid(sum);
        }

        public void SetValue(double? value)
        {
            _output = value;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}