using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using System;

namespace ConsoleApp1
{
    class Program
    {
        const string _dataPath = //@"C:\Code\ML\ConsoleApp1\Test.csv";
        @"C:\Code\ML\ConsoleApp1\TestData\Flight Delays Data.csv";
        const string _evaluatePath = @"C:\Code\ML\ConsoleApp1\TestData\Test.csv";
        const string _modelPath = @"C:\Code\ML\ConsoleApp1\model.zip";

        static void Main(string[] args)
        {
            try
            {
                //var text = System.IO.File.ReadAllLines(_dataPath);

                var pipeline = new LearningPipeline();
                pipeline.Add(new TextLoader<FlightData>(_dataPath, useHeader: true, separator: ","));
                pipeline.Add(new ColumnCopier(("DepDel15", "Label")));
                pipeline.Add(new MissingValuesRowDropper()
                {
                    Column = new string[]
                     {
                         "Year",
                         "Month",
                         "DayofMonth",
                         "DayOfWeek",
                         "Carrier",
                         "OriginAirportID",
                         "DestAirportID",
                         "CRSDepTime",
                         "DepDelay",
                         "DepDel15",
                         "CRSArrTime",
                         "ArrDelay",
                         "ArrDel15",
                         "Cancelled"
                     }
                });

                pipeline.Add(new CategoricalOneHotVectorizer( 
                         "Month",
                         "DayofMonth",
                         "DayOfWeek",
                      //   "Carrier",
                         "OriginAirportID",
                         "DestAirportID"//,
                         //"CRSDepTime"
                         ));

                pipeline.Add(new ColumnConcatenator("Features",                       
                         "Month",
                         "DayofMonth",
                         "DayOfWeek",
                         //"Carrier",
                         "OriginAirportID",
                         "DestAirportID"//,
                          //"CRSDepTime"
                         //"DepDelay"
                         ));

                pipeline.Add(new FastTreeBinaryClassifier()
                {
                      UnbalancedSets = true
                });

                var model = pipeline.Train<FlightData, FlightDelayPredictor>();

                model.WriteAsync(_modelPath);

                var testData = new TextLoader<FlightData>(_dataPath, useHeader: true, separator: ",");
                var evaluator = new BinaryClassificationEvaluator();
                BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
                //model.Predict(new FlightData() { })
                model.WriteAsync(_modelPath);


                var testRows = System.IO.File.ReadAllLines(_dataPath);

                foreach (var testRow in testRows)
                {
                    var objects = testRow.Split(",");
                    var obj = new FlightData()
                    {
                        Month = objects[1],
                        DayofMonth = objects[2],
                        DayOfWeek = objects[3],
                        OriginAirportID = objects[5],
                        DestAirportID = objects[6],
                        CRSDepTime = objects[7]
                    };

                    var prediction = model.Predict(obj);

                    if (prediction.DepDel15 != 0)
                        Console.WriteLine("Predicted DepartureDelay is: {0}", prediction.DepDel15);
                }


                //var prediction = model.Predict(new FlightData()
                //{ //2013,6,30,7,AA,13930,10721,1855,64.00,1.00,2215,45.00,1.00,0.00

                //    Year = "2013", 
                //     Month = "6",
                //     DayofMonth = "30",
                //     DayOfWeek = "7",
                //     Carrier = "AA",
                //     OriginAirportID = "13930",
                //     DestAirportID = "10721",
                //     CRSDepTime = "1855",
                //     DepDelay =64.00f,
                //     //DepDel15 = 1.00f,
                //     CRSArrTime = 2215,
                //     ArrDel15 = 1.00f,
                //     Cancelled = 0.00f

                //}
                //);
                //Console.WriteLine("Predicted DepartureDelay is: {0}", prediction.DepDel15);

            }
            catch (Exception ex)
            {

            }

        }
    }

    public class FlightDelayPredictor
    {
        [Column("Score")]
        public float DepDel15;
    }

    public class FlightData
    {
        [Column(ordinal: "0")]
        public string Year ;

        [Column(ordinal: "1")]
        public string Month ;

        [Column(ordinal: "2")]
        public string DayofMonth ;

        [Column(ordinal: "3")]
        public string DayOfWeek ;

        [Column(ordinal: "4")]
        public string Carrier;

        [Column(ordinal: "5")]
        public string OriginAirportID;

        [Column(ordinal: "6")]
        public string DestAirportID;

        [Column(ordinal: "7")]
        public string CRSDepTime;

        [Column(ordinal: "8")]
        public float DepDelay;

        [Column(ordinal: "9")]
        public float DepDel15;

        [Column(ordinal: "10")]
        public float CRSArrTime;

        [Column(ordinal: "11")]
        public float ArrDelay;

        [Column(ordinal: "12")]
        public float ArrDel15;

        [Column(ordinal: "13")]
        public float Cancelled;
    }
}

//Year,Month,DayofMonth,DayOfWeek,Carrier,OriginAirportID,DestAirportID,CRSDepTime,DepDelay,DepDel15,CRSArrTime,ArrDelay,ArrDel15,Cancelled
//2013,10,16,3,B6,14524,10721,1840,-6.00,0.00,2010,-21.00,0.00,0.00
//2013,10,15,2,AA,12953,13930,0940,-8.00,0.00,1120,-33.00,0.00,0.00
//2013,10,7,1,B6,12478,14831,1850,0.00,0.00,2212,-10.00,0.00,0.00
//2013,10,7,1,B6,14831,12478,2305,-15.00,0.00,0728,-32.00,0.00,0.00
//2013,6,24,1,WN,14747,11292,1440,10.00,0.00,1825,5.00,0.00,0.00
//2013,6,24,1,WN,14747,12191,1250,1.00,0.00,1915,-14.00,0.00,0.00