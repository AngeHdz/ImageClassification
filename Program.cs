using Microsoft.ML;
using Microsoft.ML.Vision;
using System;
using System.Linq;

MLContext mlContext = new MLContext();

string PathTraining = "C:\\Avance4\\Modelo01\\ImageClassification\\Train\\";

IDataView trainData = LoadImageFromFolder(mlContext, PathTraining);

// Data process configuration with pipeline data transformations
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: @"Label", inputColumnName: @"Label", addKeyValueAnnotationsAsText: false)
                        .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(labelColumnName: @"Label", scoreColumnName: @"Score", featureColumnName: @"ImageSource"))
                        .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: @"PredictedLabel", inputColumnName: @"PredictedLabel"));

//var pipeline = preprocessingPipeline
//    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "softmax2"))
//    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//var trainTestData = mlContext.Data.TrainTestSplit(Dta, testFraction: 0.2);
//var trainData = trainTestData.TrainSet;
//var testData = trainTestData.TestSet;


var model = pipeline.Fit(trainData);

var predictions = model.Transform(trainData);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "PredictedLabel");

Console.WriteLine($"Log-loss: {metrics.LogLoss}");

 static IDataView LoadImageFromFolder(MLContext mlContext, string folder)
{
    var res = new List<ModelInput>();
    var allowedImageExtensions = new[] { ".png", ".jpg", ".jpeg", ".gif" };
    DirectoryInfo rootDirectoryInfo = new DirectoryInfo(folder);
    DirectoryInfo[] subDirectories = rootDirectoryInfo.GetDirectories();

    if (subDirectories.Length == 0)
    {
        throw new Exception("fail to find subdirectories");
    }

    foreach (DirectoryInfo directory in subDirectories)
    {
        var imageList = directory.EnumerateFiles().Where(f => allowedImageExtensions.Contains(f.Extension.ToLower()));
        if (imageList.Count() > 0)
        {
            res.AddRange(imageList.Select(i => new ModelInput
            {
                Label = directory.Name,
                ImageSource = File.ReadAllBytes(i.FullName),
            }));
        }
    }
    return mlContext.Data.LoadFromEnumerable(res);
}