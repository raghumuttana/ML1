//using Microsoft.ML.Models;
//using Microsoft.ML.Runtime;
//using Microsoft.ML.Runtime.Api;
//using Microsoft.ML.Trainers;
//using Microsoft.ML.Transforms;
//using System.Collections.Generic;
//using System.Linq;
//using Microsoft.ML;
//using System;

//namespace ConsoleApp1
//{
//    class Program2
//    {
//        const string _dataPath = //@"C:\Code\ML\ConsoleApp1\Test.csv";
//            @"C:\Code\MachineLearning\TestData\April2017.csv";

//        const string _evaluatePath = @"C:\Code\MachineLearning\TestData\April2017.csv";
//        const string _modelPath = @"C:\Code\MachineLearning\ConsoleApp1\model.zip";

//        static void Main2(string[] args)
//        {
//            try
//            {
//                //"MONTH","DAY_OF_MONTH","DAY_OF_WEEK","CARRIER","ORIGIN","DEST","CRS_DEP_TIME","DEP_TIME","DEP_DEL15"

//                var pipeline = new LearningPipeline();
//                pipeline.Add(new TextLoader<FlightData>(_dataPath, useHeader: true, separator: ","));
//                pipeline.Add(new ColumnCopier(("DEP_DEL15", "Label")));
//                pipeline.Add(new MissingValuesRowDropper()
//                {
//                    Column = new string[]
//                     {
//                        "MONTH","DAY_OF_MONTH","DAY_OF_WEEK","AIRLINE_ID","CARRIER","ORIGIN_AIRPORT_ID","ORIGIN","DEST_AIRPORT_ID","DEST","CRS_DEP_TIME","DEP_TIME","DEP_DEL15"
//                     }
//                });

//                pipeline.Add(new CategoricalOneHotVectorizer(
//                          "MONTH",
//                         "DAY_OF_MONTH",
//                         "DAY_OF_WEEK",
//                         "AIRLINE_ID",
//                         "ORIGIN_AIRPORT_ID",
//                         "DEST_AIRPORT_ID"
//                         ));

//                pipeline.Add(new ColumnConcatenator("Features",
//                         "MONTH",
//                         "DAY_OF_MONTH",
//                         "DAY_OF_WEEK",
//                         "AIRLINE_ID",
//                         "ORIGIN_AIRPORT_ID",
//                         "DEST_AIRPORT_ID"
//                         ));

//                pipeline.Add(new FastTreeBinaryClassifier()
//                {
//                    UnbalancedSets = true
//                });

//                var model = pipeline.Train<FlightData, FlightDelayPredictor>();

//                model.WriteAsync(_modelPath);

//                var testData = new TextLoader<FlightData>(_evaluatePath, useHeader: true, separator: ",");
//                var evaluator = new BinaryClassificationEvaluator();
//                BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
//                //model.Predict(new FlightData() { })
//                model.WriteAsync(_modelPath);

//                // var testRows = System.IO.File.ReadAllLines(@"C:\Code\MachineLearning\TestData\March2017.csv");

//                //foreach(var testRow in testRows)
//                // {
//                //var objects = testRow.Split(",");
//                //var obj = new FlightData()
//                //{
//                //    MONTH = objects[0],
//                //    DAY_OF_MONTH = objects[1],
//                //    DAY_OF_WEEK = objects[2],
//                //    AIRLINE_ID = objects[3],
//                //    ORIGIN_AIRPORT_ID = objects[5],
//                //    DEST_AIRPORT_ID = objects[7]

//                //};
//                //var prediction = model.Predict(obj);

//                //if (prediction.DEP_DEL15 > 0)
//                //    Console.WriteLine("Prediction : {0} Actual : {1}", model.Predict(obj).DEP_DEL15, objects[8]);
//                // }

//            }
//            catch (Exception ex)
//            {

//            }

//        }
//    }

//    public class FlightDelayPredictor
//    {
//        [Column("Score")]
//        public float DEP_DEL15;
//    }
//    ////"MONTH","DAY_OF_MONTH","DAY_OF_WEEK","AIRLINE_ID","CARRIER","ORIGIN_AIRPORT_ID","ORIGIN","DEST_AIRPORT_ID","DEST","CRS_DEP_TIME","DEP_TIME","DEP_DEL15",
//    public class FlightData
//    {
//        [Column(ordinal: "0")]
//        public string MONTH;

//        [Column(ordinal: "1")]
//        public string DAY_OF_MONTH;

//        [Column(ordinal: "2")]
//        public string DAY_OF_WEEK;

//        [Column(ordinal: "3")]
//        public string AIRLINE_ID;

//        [Column(ordinal: "4")]
//        public string CARRIER;

//        [Column(ordinal: "5")]
//        public string ORIGIN_AIRPORT_ID;

//        [Column(ordinal: "6")]
//        public string ORIGIN;

//        [Column(ordinal: "7")]
//        public string DEST_AIRPORT_ID;

//        [Column(ordinal: "8")]
//        public string DEST;

//        [Column(ordinal: "9")]
//        public string CRS_DEP_TIME;

//        [Column(ordinal: "10")]
//        public string DEP_TIME;

//        [Column(ordinal: "11")]
//        public float DEP_DEL15;
//    }
//    // //"MONTH","DAY_OF_MONTH","DAY_OF_WEEK","CARRIER","ORIGIN","DEST","CRS_DEP_TIME","DEP_TIME","DEP_DEL15"
//    //public class FlightData
//    //{
//    //    [Column(ordinal: "0")]
//    //    public string MONTH;

//    //    [Column(ordinal: "1")]
//    //    public string DAY_OF_MONTH;

//    //    [Column(ordinal: "2")]
//    //    public string DAY_OF_WEEK;

//    //    [Column(ordinal: "3")]
//    //    public string CARRIER;

//    //    [Column(ordinal: "4")]
//    //    public string ORIGIN;

//    //    [Column(ordinal: "5")]
//    //    public string DEST;

//    //    [Column(ordinal: "6")]
//    //    public string CRS_DEP_TIME;

//    //    [Column(ordinal: "7")]
//    //    public string DEP_TIME;

//    //    [Column(ordinal: "8")]
//    //    public float DEP_DEL15;
//    //}
//}

////Year,Month,DayofMonth,DayOfWeek,Carrier,OriginAirportID,DestAirportID,CRSDepTime,DepDelay,DepDel15,CRSArrTime,ArrDelay,ArrDel15,Cancelled
////2013,10,16,3,B6,14524,10721,1840,-6.00,0.00,2010,-21,0.00,0.00
////2013,10,15,2,AA,12953,13930,0940,-8.00,0.00,1120,-33.00,0.00,0.00
////2013,10,7,1,B6,12478,14831,1850,0.00,0.00,2212,-10.00,0.00,0.00
////2013,10,7,1,B6,14831,12478,2305,-15.00,0.00,0728,-32.00,0.00,0.00
////2013,6,24,1,WN,14747,11292,1440,10.00,0.00,1825,5.00,0.00,0.00
////2013,6,24,1,WN,14747,12191,1250,1,0.00,1915,-14.00,0.00,0.00

