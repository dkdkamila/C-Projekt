using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SwearWordDetection
{
    class SwearData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }

    class SwearPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }

    class Program
    {
        static List<string> posts = new List<string>();
        static MLContext mlContext = new MLContext();
        static PredictionEngine<SwearData, SwearPrediction> predictionEngine;

        static void Main(string[] args)
        {
            var data = mlContext.Data.LoadFromTextFile<SwearData>("swear_data.csv", hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SwearData.Text))
                            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
            var model = pipeline.Fit(data);
            predictionEngine = mlContext.Model.CreatePredictionEngine<SwearData, SwearPrediction>(model);

            bool appRunning = true;

            while (appRunning)
            {
                Console.WriteLine("Välkommen till appen! Vad vill du göra?");
                Console.WriteLine("1. Skriv ett inlägg");
                Console.WriteLine("2. Visa tidigare inlägg");
                Console.WriteLine("3. Radera inlägg");
                Console.WriteLine("4. Avsluta");

                string choice = Console.ReadLine();

                switch (choice)
                {
                    case "1":
                        CreatePost();
                        break;
                    case "2":
                        ViewPosts();
                        break;
                    case "3":
                        DeletePost();
                        break;
                    case "4":
                        appRunning = false;
                        Console.WriteLine("Tack för att du använde appen. Hej då!");
                        break;
                    default:
                        Console.WriteLine("Ogiltigt val. Försök igen.");
                        break;
                }
            }
        }

        static void CreatePost()
        {
            Console.WriteLine("Skriv ditt inlägg:");
            string userPost = Console.ReadLine();

            if (ContainsBannedWords(userPost))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Inlägget innehåller svärord och kan inte publiceras. Var vänlig och skriv ett nytt inlägg.");
                Console.ForegroundColor = ConsoleColor.Gray;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
                posts.Add(userPost);
                Console.WriteLine("Inlägget har publicerats.");
                Console.ForegroundColor = ConsoleColor.Gray;
            }
        }

        static void ViewPosts()
        {
            Console.WriteLine("Tidigare inlägg:");
            if (posts.Count == 0)
            {
                Console.WriteLine("Inga tidigare inlägg.");
            }
            else
            {
                for (int i = 0; i < posts.Count; i++)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    Console.WriteLine($"{i + 1}. {posts[i]}");
                    Console.ForegroundColor = ConsoleColor.Gray;

                }
            }


        }

        static void DeletePost()
        {
            Console.WriteLine("Vilket inlägg vill du radera? Ange numret:");
            ViewPosts();
            if (int.TryParse(Console.ReadLine(), out int index))
            {
                if (index > 0 && index <= posts.Count)
                {
                    posts.RemoveAt(index - 1);
                    Console.WriteLine("Inlägget har raderats.");
                }
                else
                {
                    Console.WriteLine("Ogiltigt inläggsnummer.");
                }
            }
            else
            {
                Console.WriteLine("Felaktig inmatning.");
            }
        }

        static bool ContainsBannedWords(string text)
        {
            var newData = new SwearData { Text = text };
            var prediction = predictionEngine.Predict(newData);
            return prediction.Prediction;
        }
    }
}

