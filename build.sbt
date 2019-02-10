name := "NBO_ph2-MBB_POSTPAID_UPSell_Model"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.slf4j" % "slf4j-api" % "1.7.10",
  "org.apache.spark" %% "spark-core" % "1.6.2",
  "org.apache.spark" %% "spark-mllib" % "1.6.2",
  "org.apache.spark" %% "spark-sql" % "1.6.2",
  "org.apache.kudu" % "kudu-client" % "1.6.0",
  "org.apache.kudu" %% "kudu-spark" % "1.5.0"
)

