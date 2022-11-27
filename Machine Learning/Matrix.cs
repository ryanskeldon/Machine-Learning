using System;
using System.Collections.Generic;
using System.Linq;

namespace Machine_Learning
{
    public class Matrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; }
        public double[,] Data { get; set; }

        public Matrix(int rows, int columns)
        {
            if (rows < 1) throw new ArgumentOutOfRangeException(nameof(rows));
            if (columns < 1) throw new ArgumentOutOfRangeException(nameof(columns));

            Rows = rows;
            Columns = columns;

            Data = new double[rows, columns];
        }

        public Matrix(double[,] data)
        {
            Rows = data.GetLength(0);
            Columns = data.GetLength(1);
            Data = data;
        }

        public void Map(Func<double, double> func)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    var value  = Data[i, j];
                    Data[i, j] = func(value);
                }
            }
        }

        public static Matrix Map(Matrix input, Func<double, double> func)
        {
            var output = new Matrix(input.Rows, input.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    var value = input.Data[i, j];
                    output.Data[i, j] = func(value);
                }
            }

            return output;
        }

        public void Add(double scalar) => Map(x => x + scalar);

        public void Add(Matrix b)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    Data[i, j] += b.Data[i, j];
                }
            }
        }

        public static Matrix Add(Matrix a, Matrix b)
        {
            var output = new Matrix(a.Rows, a.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    output.Data[i, j] = a.Data[i, j] + b.Data[i, j];
                }
            }

            return output;
        }

        public void Subtract(double scalar) => Map(x => x - scalar);

        public void Subtract(Matrix b)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    Data[i, j] -= b.Data[i, j];
                }
            }

        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            var output = new Matrix(a.Rows, a.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    output.Data[i, j] = a.Data[i, j] - b.Data[i, j];
                }
            }

            return output;
        }

        public void Multiply(double scalar) => Map(x => x * scalar);

        public void Multiply(Matrix b)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    Data[i, j] *= b.Data[i, j];
                }
            }
        }

        public static Matrix Multiply(Matrix a, Matrix b)
        {
            var output = new Matrix(a.Rows, a.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    output.Data[i, j] = a.Data[i, j] * b.Data[i, j];
                }
            }

            return output;
        }

        public void Divide(double scalar) => Map(x => x / scalar);

        public void Divide(Matrix b)
        {
            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    Data[i, j] /= b.Data[i, j];
                }
            }
        }

        public static Matrix Divide(Matrix a, Matrix b)
        {
            var output = new Matrix(a.Rows, a.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    output.Data[i, j] = a.Data[i, j] / b.Data[i, j];
                }
            }

            return output;
        }

        public Matrix Transpose()
        {
            var output = new Matrix(Columns, Rows);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    output.Data[i, j] = Data[j, i];
                }
            }

            return output;

        }

        public Matrix Copy()
        {
            var output = new Matrix(Rows, Columns);

            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    output.Data[i, j] = Data[i, j];
                }
            }

            return output;
        }

        public static Matrix DotProduct(Matrix a, Matrix b)
        {
            var output = new Matrix(a.Rows, b.Columns);

            for (var i = 0; i < output.Rows; i++)
            {
                for (var j = 0; j < output.Columns; j++)
                {
                    for (var k = 0; k < a.Columns; k++)
                    {
                        output.Data[i, j] += a.Data[i, k] * b.Data[k, j];
                    }
                }
            };

            return output;
        }

        public double[] ToArray()
        {
            var output = new List<double>();

            for (var i = 0; i < Rows; i++)
            {
                for (var j = 0; j < Columns; j++)
                {
                    output.Add(Data[i, j]);
                }
            }

            return output.ToArray();
        }

        public static Matrix FromArray(double[] input)
        {
            var output = new Matrix(input.Length, 1);

            for (var i = 0; i < input.Length; i++)
            {
                output.Data[i, 0] = input[i];
            }

            return output;
        }

        public string Serialize()
        {
            return $"{Rows},{Columns},{string.Join("|", ToArray())}";
        }

        public static Matrix Deserialize(string input)
        {
            var pieces = input.Split(',');
            var matrix = new Matrix(int.Parse(pieces[0]), int.Parse(pieces[1]));
            var flatData = pieces[2].Split('|').Select(double.Parse).ToArray();

            var data = new double[matrix.Rows, matrix.Columns];

            for (var i = 0; i < matrix.Rows; i++)
            {
                for (var j = 0; j < matrix.Columns; j++)
                {
                    var x = i * matrix.Columns + j;
                    data[i, j] = flatData[x];
                }
            }

            matrix.Data = data;

            return matrix;
        }
    }
}