using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace RestaurantRecommender.AI
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public string UserId { get; set; }
        [LoadColumn(1)]
        public string RestaurantName { get; set; }
        [LoadColumn(2)]
        public float TotalRating { get; set; }
    }
}
