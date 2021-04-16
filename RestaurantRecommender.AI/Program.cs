using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RestaurantRecommender.AI.Helper;
using SHA.BeautifulConsoleColor.Core.Class;
using SHA.BeautifulConsoleColor.Core.Model;
using MLContext = Microsoft.ML.MLContext;

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

			var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(options);

			var trainerPipeLine = dataPreProcessingPipeLine.Append(trainer);
            BCCConsole.Write(BCCConsoleColor.DarkBlue,false,"\n","Training Model");
            var model = trainerPipeLine.Fit(trainingDataView);

            //Test 
            var testUserId = "U1134";
            var predictionEngine = mlContext.Model
	            .CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var alreadyRatedRestaurant = mlContext.Data
	            .CreateEnumerable<ModelInput>(trainingDataView, false)
	            .Where(r => r.UserId == testUserId)
	            .Select(r => r.RestaurantName)
	            .Distinct();
            var allRestaurantNames = trainingDataView
	            .GetColumn<string>("RestaurantName")
	            .Distinct().Where(r => !alreadyRatedRestaurant.Contains(r));
            var scoredRestaurant = allRestaurantNames
	            .Select(rn =>
	            {
					var prediction = predictionEngine.Predict(
						new ModelInput()
						{
                            UserId = testUserId,
                            RestaurantName = rn
						});
					return (RestaurantName: rn, PredictedScore: prediction.Score);
	            });

            var top10Restaurant = scoredRestaurant
	            .OrderByDescending(r => r.PredictedScore)
	            .Take(10);
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false,
	            "\n",
	            $"Top 10 Restaurant Name & Rate For User {testUserId}",
	            "----------------------------------------------------");
            foreach (var top in top10Restaurant)
            {
	            BCCConsole.Write(BCCConsoleColor.DarkGreen,false,$"Prediction Score [{top.PredictedScore:#.0}] | Restaurant Name [{top.RestaurantName}] ");
            }
            BCCConsole.Write(BCCConsoleColor.DarkGreen,false, "----------------------------------------------------");
        }
    }
}
