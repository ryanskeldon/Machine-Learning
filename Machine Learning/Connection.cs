namespace Machine_Learning
{
    public class Connection
    {
        public double Weight { get; set; }
        public Node FromNode { get; }
        public Node ToNode { get; }
        public string Id => $"{FromNode.Id}->{ToNode.Id}";

        public Connection(Node fromNode, Node toNode, double weight)
        {
            FromNode = fromNode;
            ToNode = toNode;
            Weight = weight;
        }
    }
}