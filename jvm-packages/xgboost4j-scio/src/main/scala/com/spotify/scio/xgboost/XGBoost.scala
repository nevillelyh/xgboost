/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package com.spotify.scio.xgboost

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, File}

import com.google.cloud.dataflow.sdk.options.PipelineOptionsFactory
import com.google.cloud.dataflow.sdk.runners.inprocess.InProcessPipelineRunner
import com.spotify.scio._
import com.spotify.scio.values.SCollection
import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker, XGBoostError}
import ml.dmlc.xgboost4j.scala.{XGBoost => SXGBoost, _}
import org.apache.commons.logging.LogFactory

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostScio")

  def train(trainingData: SCollection[LabeledPoint],
            configMap: Map[String, Any], round: Int, nWorkers: Int,
            obj: ObjectiveTrait = null, eval: EvalTrait = null,
            useExternalMemory: Boolean = false): SCollection[Array[Byte]] = {
    val sc = trainingData.context
//    val opts = sc.optionsAs[DataflowPipelineOptions]
//    require(
//      opts.getAutoscalingAlgorithm == AutoscalingAlgorithmType.NONE,
//      "XGBoost can not be used with autoscaling")
//    val nWorkers = opts.getNumWorkers
    require(nWorkers > 0, "nWorkers must be greater than 0")
    val xgBoostConfMap = configMap + ("nthread" -> 1)

    val rabitEnv = sc.parallelize(Seq(nWorkers))
      .map { n =>
        logger.info("========== Loading worker environment")
        val tracker = new RabitTracker(n)
        require(tracker.start(), "Failed to start tracker")
        val env = tracker.getWorkerEnvs.asScala
        tracker.stop()
        logger.info("========== Worker environement: " + env)
        env
      }

    trainingData
      .map((Random.nextInt(nWorkers), _))
      .groupByKey  // id, trainingSamples
      .union(sc.parallelize(Seq((nWorkers, Iterable.empty))))
      .cross(rabitEnv)
      .map { case ((id, trainingSamples), env) =>
        if (id == nWorkers) {
          // master node
          logger.info("========== Starting master")
          val tracker = new RabitTracker(nWorkers)
          require(tracker.start(), "Failed to start tracker")
          val returnVal = tracker.waitFor()
          if (returnVal == 0) {
            Array.emptyByteArray
          } else {
            logger.info("========== XGBoostModel training failed " + returnVal)
            throw new XGBoostError("XGBoostModel training failed")
          }
        } else {
          // worker node
          val initEnv = env + ("DMLC_TASK_ID" -> id.toString)
          logger.info("========== Starting worker " + id + " " + initEnv)
          Rabit.init(initEnv.asJava)
          val iter = trainingSamples.iterator
          if (iter.hasNext) {
            logger.info("========== Starting training on worker " + id)
            val cacheFileName: String = if (useExternalMemory) {
              s"xgboost-dtrain-cache-$id"
            } else {
              null
            }
            val trainingSet = new DMatrix(iter, cacheFileName)
            val booster = SXGBoost.train(trainingSet, xgBoostConfMap, round,
              watches = new mutable.HashMap[String, DMatrix] {
                put("train", trainingSet)
              }.toMap, obj, eval)
            Rabit.shutdown()
            logger.info("========== Training on worker " + id + " done " + booster)
            val bos = new ByteArrayOutputStream()
            booster.saveModel(bos)
            bos.toByteArray
          } else {
            logger.info("========== Empty bundle in training dataset")
            Rabit.shutdown()
            throw new XGBoostError("Empty bundle in training dataset")
          }
        }
      }
      .filter(_.nonEmpty)
      .reduce((x, y) => x)
  }

  def main(args: Array[String]): Unit = {
//    val tracker = new RabitTracker(10)
//    tracker.start()
//    tracker.getWorkerEnvs.asScala.foreach(println)
//    tracker.stop()

    val p = PipelineOptionsFactory.create()
    p.setRunner(classOf[InProcessPipelineRunner])
    val sc = ScioContext(p)

    val path = "/home/neville/src/gcp/xgboost/jvm-packages/xgboost4j-spark/src/test/resources"
    val trainingSet = readFile(path + "/agaricus.txt.train")
    val trainingData = sc.parallelize(trainingSet)
    val paramMap = Map("eta" -> "1", "max_depth" -> "2", "silent" -> "0",
      "objective" -> "binary:logistic")
    val f = XGBoost.train(trainingData, paramMap, 5, 2).materialize
    sc.close()

    logger.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    val bytes = f.waitForResult().value.next()
    val booster = SXGBoost.loadModel(new ByteArrayInputStream(bytes))
    booster.getModelDump().foreach(println)
  }

  private def readFile(filePath: String): List[LabeledPoint] = {
    val file = Source.fromFile(new File(filePath))
    val sampleList = new ListBuffer[LabeledPoint]
    for (sample <- file.getLines()) {
      sampleList += fromSVMStringToLabeledPoint(sample)
    }
    sampleList.toList
  }
  private def fromSVMStringToLabeledPoint(line: String): LabeledPoint = {
    val labelAndFeatures = line.split(" ")
    val label = labelAndFeatures(0).toFloat
    val features = labelAndFeatures.tail
    val denseFeature = new Array[Float](129)
    for (feature <- features) {
      val idAndValue = feature.split(":")
      denseFeature(idAndValue(0).toInt) = idAndValue(1).toFloat
    }
    LabeledPoint.fromDenseVector(label, denseFeature)
  }
}
