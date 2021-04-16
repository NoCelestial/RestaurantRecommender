using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using RestaurantRecommender.AI.Helper;
using SHA.BeautifulConsoleColor.Core.Class;
using SHA.BeautifulConsoleColor.Core.Model;

namespace RestaurantRecommender.AI
{
    class Program
    {
        static void Main(string[] args)
        {
            BCCConsole.Write(BCCConsoleColor.DarkBlue,false,"Restaurant Recommender Is Started . . .");

            MLContext mlContext = new MLContext(0);
            var trainingDataFile = Environment.CurrentDirectory + @"\Data\TrainingFile.tsv";
            DataPreparer.PreprocessData(trainingDataFile);
            IDataView trainingDataView = mlContext.Data
                .LoadFromTextFile<ModelInput>(trainingDataFile,hasHeader:true);

            var dataPreProcessingPipeLine = mlContext.Transforms.Conversion
                .MapValueToKey("UserIdEncoded",nameof(ModelInput.UserId))
                .Append(mlContext.Transforms.Conversion
                    .MapValueToKey("RestaurantNameEncoded",nameof(ModelInput.RestaurantName)));

			var options = new MatrixFactorizationTrainer.Options
			{
				MatrixColumnIndexColumnName = "UserIdEncoded",
				MatrixRowIndexColumnName = "RestaurantNameEncoded",
				LabelColumnName = "TotalRating",
				NumberOfIterations = 100,
				ApproximationRank = 100,
                Quiet = true
			};

		}
    }
}
