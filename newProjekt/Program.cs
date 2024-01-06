using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SwearWordDetection
{
    // Klassen representerar datamodellen för att lagra text och dess svärordsmärkning
    class SwearData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1)]
        public bool Label { get; set; }
    }
    // Klassen representerar förutsägelsemodellen för att lagra svärordsförutsägelser
    class SwearPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }

    class Program
    {
        // Lista som lagrar användarinlägg
        static List<string> posts = new List<string>();
        // Skapar ett MLContext-objekt för ML.NET-användning
        static MLContext mlContext = new MLContext();
        // En motor för förutsägelse som använder tränad modell
        static PredictionEngine<SwearData, SwearPrediction> predictionEngine;
        // Huvudmetod för att köra applikationen
        static void Main(string[] args)
        {
            // Laddar in träningsdata från en CSV-fil
            var data = mlContext.Data.LoadFromTextFile<SwearData>("swear_data.csv", hasHeader: true, separatorChar: ',');
            // Bygger en pipeline för att förbereda data och skapa en klassificeringsmodell
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SwearData.Text))
                            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
            // Tränar modellen med inläst data
            var model = pipeline.Fit(data);
            // Skapar en förutsägelsemotor baserad på den tränade modellen
            predictionEngine = mlContext.Model.CreatePredictionEngine<SwearData, SwearPrediction>(model);

            bool appRunning = true;
            // Loop som hanterar användarinteraktionen
            while (appRunning)
            {
                // Presententation av användaralternativ
                Console.WriteLine("Välkommen till appen! Vad vill du göra?");
                Console.WriteLine("1. Skriv ett inlägg");
                Console.WriteLine("2. Visa tidigare inlägg");
                Console.WriteLine("3. Radera inlägg");
                Console.WriteLine("4. Avsluta");

                string choice = Console.ReadLine();
                // Väljer en åtgärd baserat på användarens val
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
        // Metod för att skapa inlägg och hantera svärordsfiltrering
        static void CreatePost()
        {
            // Läser in användarinput för ett inlägg
            Console.WriteLine("Skriv ditt inlägg:");
            string userPost = Console.ReadLine();
            // Kontrollerar om inlägget innehåller svärord
            if (ContainsBannedWords(userPost))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Inlägget innehåller svärord och kan inte publiceras. Var vänlig och skriv ett nytt inlägg.");
                Console.ForegroundColor = ConsoleColor.Gray;
            }
            else
            {
                // Publicerar inlägget om det inte innehåller svärord
                Console.ForegroundColor = ConsoleColor.Green;
                posts.Add(userPost);
                Console.WriteLine("Inlägget har publicerats.");
                Console.ForegroundColor = ConsoleColor.Gray;
            }
        }

        // Metod för att visa tidigare inlägg
        static void ViewPosts()
        {
            // Kontrollerar om det finns tidigare inlägg att visa
            Console.WriteLine("Tidigare inlägg:");
            if (posts.Count == 0)
            {
                Console.WriteLine("Inga tidigare inlägg.");
            }
            else
            {
                // Visar tidigare inlägg och ändrar textfärgen till blå
                for (int i = 0; i < posts.Count; i++)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    Console.WriteLine($"{i + 1}. {posts[i]}");
                    Console.ForegroundColor = ConsoleColor.Gray;

                }
            }


        }

        // Metod för att radera tidigare inlägg
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
        // Metod för att bedöma om inlägget innehåller svärord
        static bool ContainsBannedWords(string text)
        {
            var newData = new SwearData { Text = text };
            var prediction = predictionEngine.Predict(newData);
            return prediction.Prediction;
        }
    }
}

