using System;
using System.Collections.Generic;
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
			BCCConsole.Write(BCCConsoleColor.DarkBlue, false, "Restaurant Recommender Is Started . . .");

			MLContext mlContext = new MLContext(0);
			var trainingDataFile = Environment.CurrentDirectory + @"\Data\TrainingFile.tsv";
			DataPreparer.PreprocessData(trainingDataFile);
			IDataView trainingDataView = mlContext.Data
				.LoadFromTextFile<ModelInput>(trainingDataFile, hasHeader: true);

			var dataPreProcessingPipeLine = mlContext.Transforms.Conversion
				.MapValueToKey("UserIdEncoded", nameof(ModelInput.UserId))
				.Append(mlContext.Transforms.Conversion
					.MapValueToKey("RestaurantNameEncoded", nameof(ModelInput.RestaurantName)));

			var options = new MatrixFactorizationTrainer.Options
			{
				MatrixColumnIndexColumnName = "UserIdEncoded",
				MatrixRowIndexColumnName = "RestaurantNameEncoded",
				LabelColumnName = "TotalRating",
				NumberOfIterations = 10,
				ApproximationRank = 200,
				Quiet = true
			};

			var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(options);

			var trainerPipeLine = dataPreProcessingPipeLine.Append(trainer);

			#region Not Using CV

			BCCConsole.Write(BCCConsoleColor.DarkBlue, false, "\n", "Training Model");
			var model = trainerPipeLine.Fit(trainingDataView);

			////Test 
			//var testUserId = "U1134";
			var predictionEngine = mlContext.Model
				.CreatePredictionEngine<ModelInput, ModelOutput>(model);
			//var alreadyRatedRestaurant = mlContext.Data
			//	.CreateEnumerable<ModelInput>(trainingDataView, false)
			//	.Where(r => r.UserId == testUserId)
			//	.Select(r => r.RestaurantName)
			//	.Distinct();
			//var allRestaurantNames = trainingDataView
			//	.GetColumn<string>("RestaurantName")
			//	.Distinct().Where(r => !alreadyRatedRestaurant.Contains(r));
			//var scoredRestaurant = allRestaurantNames
			//	.Select(rn =>
			//	{
			//		var prediction = predictionEngine.Predict(
			//			new ModelInput()
			//			{
			//				UserId = testUserId,
			//				RestaurantName = rn
			//			});
			//		return (RestaurantName: rn, PredictedScore: prediction.Score);
			//	});

			//var top10Restaurant = scoredRestaurant
			//	.OrderByDescending(r => r.PredictedScore)
			//	.Take(10);
			//BCCConsole.Write(BCCConsoleColor.DarkGreen,false,
			//	"\n",
			//	$"Top 10 Restaurant Name & Rate For User {testUserId}",
			//	"----------------------------------------------------");
			//foreach (var top in top10Restaurant)
			//{
			//	BCCConsole.Write(BCCConsoleColor.DarkGreen,false,$"Prediction Score [{top.PredictedScore:#.0}] | Restaurant Name [{top.RestaurantName}] ");
			//}
			//BCCConsole.Write(BCCConsoleColor.DarkGreen,false, "----------------------------------------------------");

			#endregion

			#region Using CV

			//var cvMetrics = mlContext.Recommendation()
			//	.CrossValidate(trainingDataView, trainerPipeLine, labelColumnName: "TotalRating");

			//var averageRMSE = cvMetrics.Average(cv => cv.Metrics.RootMeanSquaredError);
			//var averageRSquared = cvMetrics.Average(cv => cv.Metrics.RSquared);
			//BCCConsole.Write(BCCConsoleColor.DarkGreen, false,
			//	"\n",
			//	"Training Result Before Cross Validation (Metrics) ",
			//	"--------------------------------------------------",
			//	$"RMSE => Root Error : {averageRMSE:#.000}",
			//	$"RSQ => RSquared : {averageRSquared:#.000}",
			//	"--------------------------------------------------");

			#endregion

			var prediction = predictionEngine.Predict(new ModelInput()
			{
				UserId = "CLONED",
				RestaurantName = "Rincon Huasteco"
			});

			BCCConsole.Write(BCCConsoleColor.Green,false,"\n",$"Prediction Result Score : {prediction.Score:#.0} For Rincon Huasteco");

			//HyperParameterExploration(mlContext, dataPreProcessingPipeLine, trainingDataView);
		}

		private static void HyperParameterExploration(MLContext mlContext
			, IEstimator<ITransformer> dataPreProcessingPipeLine
			, IDataView trainDataView)
		{
			var result = new List<(double RMSE
				, double RSQ
				, int iterations
				, int approximationRank)>();
			for (int iterations = 5; iterations < 100; iterations += 5)
			{
				for (int approximationRank = 50; approximationRank < 250; approximationRank += 50)
				{
					var option = new MatrixFactorizationTrainer.Options
					{
						MatrixColumnIndexColumnName = "UserIdEncoded",
						MatrixRowIndexColumnName = "RestaurantNameEncoded",
						LabelColumnName = "TotalRating",
						NumberOfIterations = iterations,
						ApproximationRank = approximationRank,
						Quiet = true
					};

					var trainer = mlContext.Recommendation()
						.Trainers.MatrixFactorization(option);
					var completePipeLine = dataPreProcessingPipeLine.Append(trainer);
					var cvMetrics = mlContext.Recommendation()
						.CrossValidate(trainDataView, completePipeLine, labelColumnName: "TotalRating");
					result.Add((
						cvMetrics.Average(cv => cv.Metrics.RootMeanSquaredError),
					cvMetrics.Average(cv => cv.Metrics.RSquared),
					iterations,
					approximationRank));
				}

			}

			BCCConsole.Write(BCCConsoleColor.DarkGreen, false, "\n", "--- Hyper Parameter Exploration Result Metrics ---");
			foreach (var res in result.OrderByDescending(r => r.RSQ))
			{
				BCCConsole.Write(BCCConsoleColor.DarkGreen, false, "\n",
					$"RSQ => RSquared : {res.RSQ:#.000}",
					$"RMSE => Root Error : {res.RMSE:#.000}",
					$"I => Iterations : {res.iterations}",
					$"AR => ApproximationRank : {res.approximationRank}"
				);
			}
			BCCConsole.Write(BCCConsoleColor.DarkGreen, false, "\n", "---------------------------------------");
		}
	}
}
