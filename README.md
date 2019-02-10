# sparkNextBestOffer-RF
This is Spark/Scala based Next Best Offer Recommendation Model which used Random Forest Classifier Algorithm

* This is supervized learning based recommendation model(Random Forest) which gives recommended products to the customers. This Spark/Scala programme reads and writes data from/into kudu tables.
* Compile Model using sbt (Goto root folder and give following commands)
  * sbt package
  * sbt compile
* Model Run
  $PARKHOME/bin/spark-submit --class nboMbbCrossModel --master yarn --driver-memory 30G --executor-memory 20G --num-executors 8 --packages org.apache.kudu:kudu-spark_2.10:1.5.0 /home/ml/NBO/jars/nbo_model_2.10-1.0.jar
