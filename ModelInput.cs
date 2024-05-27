using Microsoft.ML;
using Microsoft.ML.Data;

public class ModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"Label")]
    public string Label { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"ImageSource")]
    public byte[] ImageSource { get; set; }


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
}