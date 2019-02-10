/*
***********NBO_ph2_MBB_Postpaid_UPSell Model******************
Created Date: 2018-07-12
Created By: hasitha_08565
Algorithm: Random Forest Classifier
**************************************************************
*/

import org.apache.kudu.spark.kudu._
import org.apache.kudu.client._
import collection.JavaConverters._

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import java.io.File
import com.typesafe.config.ConfigFactory


object nboMbbOUpModel {
	Logger.getLogger("org").setLevel(Level.WARN)
	val filePath = new File("").getAbsolutePath
	val config = ConfigFactory.parseFile(new File(filePath + "/../conf/NBO_ph2.conf"))

	val conf = new SparkConf().setAppName("NBO-MBB_POSTPAID_UPSell").set("spark.driver.allowMultipleContexts", "true")
	val sc = new org.apache.spark.SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  def main(args: Array[String]) = {

	println("*****************NBO-MBB_POSTPAID_UPSell Model - Random Forest Classifier******************")
	val dataset = sqlContext.read.options(Map("kudu.master" -> "10.48.30.122:7051,10.48.30.130:7051","kudu.table" -> "impala::kudu_tabs.nbo_mbb_o_up_train")).kudu
	val col = dataset.select("pcrf_peak_usage","pcrf_offpeak_usage","callduration","smscount","province","average_monthly_bill_amount","network_stay","credit_category","credit_type","customer_priority_type","age","gender","preferred_language","number_of_dbn_accounts","cx_grading","lte_flag","movie","music","profession","media_response","overseas_travellers","ott_apps_user","online_shopper","selfcare_app_user","youtube","netflix").columns
		
	val assembler = vectorAssembler(dataset,col)
    assembler.printSchema()
    
    //Split training/testing
    val splits = normalize(assembler.na.drop).randomSplit(Array(0.7,0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    
    val labelIndexer = new StringIndexer().setInputCol("package_name").setOutputCol("indexedLabel").fit(normalize(assembler.na.drop))
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer,featureIndex(normalize(assembler)),randomForest,labelConverter))

    //Model training
	println("****************Model Training....************************")
    val model = pipeline.fit(train)
    
    //Model testing
	println("****************Model Testing....************************")
    val predictions = model.transform(test)
    
    //Model evaluation
	println("****************Model Evalating....************************")
    val precision = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("weightedPrecision")
    val recall = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("weightedRecall")
    val f1 = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("f1")
        
    //Cross Validation
	println("****************Cross Validating.... - K-fold cross validation************************")
    val paramGrid = new ParamGridBuilder().build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(recall)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
    
    val cvModel = cv.fit(train)
    val cvPredictions = cvModel.transform(test)
    
    println("**************Model Evaluation-Check best accuracy (After Cross Validation)******************************")
    println("**********Best fit Precision after Cross validation = " + precision.evaluate(cvPredictions) + "**********")
    println("**********Best fit Recall after Cross validation= " + recall.evaluate(cvPredictions) + "**********")
    println("**********Best fit F1 after Cross validation= " + f1.evaluate(cvPredictions) + "**********")
    println("*********************************************************************************************************")
		
	//Save Model
	println("****************Model Saving....************************")
	sc.parallelize(Seq(cvModel), 1).saveAsObjectFile(config.getString("pathPosUp"))
    //sc.parallelize(Seq(cvModel), 1).saveAsObjectFile("hdfs://nameservice1//source/NBO_ph2/pos_gsmupsell")
    println("****************GSM Upsell Postpaid Model Saved in: " + config.getString("pathPosUp") + "***********")
    
	}
	
  //***********Feature vactor
  def vectorAssembler (dataset: DataFrame, col: Array[String]) : DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(col)
      .setOutputCol("features")
    val feature = assembler.transform(dataset.na.drop)
    return feature
  }
  
  //***********Normalization
  def normalize(dataset: DataFrame): DataFrame = {
    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setMax(1)
      .setMin(0)
      .fit(dataset)
      .transform(dataset)

    return scaler
  }

  //***************Feature Indexer
  def featureIndex(dataset: DataFrame): VectorIndexerModel = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeature")
      .setMaxCategories(5)
      .fit(dataset)

    return featureIndexer
  }

  //**************Random Forest Model Building
  def randomForest(): RandomForestClassifier = {
    //Random Forest Model
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeature")
      .setImpurity("gini")
      .setMaxDepth(30)
      .setNumTrees(config.getInt("NumTreesPosUp"))
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    return rf
  }

  //****************Model Evaluation - Precision
  def modelEvaluationPrecision(): MulticlassClassificationEvaluator = {
    val precision = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    return precision
  }

  //***************Model Evaluation - Recall
  def modelEvaluationRecall(): MulticlassClassificationEvaluator = {
    val recall = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")

    return recall
  }
  
}
