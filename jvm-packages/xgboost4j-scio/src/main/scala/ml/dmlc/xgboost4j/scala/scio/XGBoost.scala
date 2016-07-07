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

package ml.dmlc.xgboost4j.scala.scio

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
import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

object XGBoost extends Serializable {
  private val logger = LogFactory.getLog("XGBoostScio")

  def train(trainingData: SCollection[LabeledPoint], configMap: Map[String, Any],
            round: Int, nWorkers: Int, obj: ObjectiveTrait = null, eval: EvalTrait = null,
            useExternalMemory: Boolean = false): SCollection[Booster] = {
    require(nWorkers > 0, "nWorkers must be greater than 0")
    val xgBoostConfMap = configMap + ("nthread" -> 1)

    val sc = trainingData.context
    val rabitEnv = sc.parallelize(Seq(nWorkers))
      .map { n =>
        logger.info("Loading XGBoost worker environment")
        val tracker = new RabitTracker(n)
        require(tracker.start(), "Failed to start tracker")
        val env = tracker.getWorkerEnvs.asScala
        tracker.stop()
        logger.info(env)
        env
      }

    trainingData
      .map((Random.nextInt(nWorkers), _))
      .union(sc.parallelize(Seq((nWorkers, LabeledPoint.fromDenseVector(0f, Array(0f))))))
      .groupByKey
      .cross(rabitEnv)
      .map { case ((id, trainingSamples), env) =>
        if (id == nWorkers) {
          // master node
          logger.info("Starting XGBoost master" + " " + Thread.currentThread().toString)
          val tracker = new RabitTracker(nWorkers)
          require(tracker.start(), "Failed to start tracker")
          val returnVal = tracker.waitFor()
          logger.info("Stopping XGBoost master")
          if (returnVal == 0) {
            Array.emptyByteArray
          } else {
            throw new XGBoostError("XGBoostModel training failed")
          }
        } else {
          // worker node
          logger.info("Starting XGBoost worker " + id + " " + Thread.currentThread().toString)
          val initEnv = env + ("DMLC_TASK_ID" -> id.toString)
          Rabit.init(initEnv.asJava)
          val iter = trainingSamples.iterator
//          val iter = readFile(path + "/agaricus.txt.train").iterator
          if (iter.hasNext) {
            val cacheFileName: String = if (useExternalMemory) {
              s"xgboost-dtrain-cache-$id"
            } else {
              null
            }
            val trainingSet = new DMatrix(iter, cacheFileName)
//            val trainingSet = new DMatrix(path + "/agaricus.txt.train")
            println(s"$xgBoostConfMap\t$round\t$nWorkers\t${Rabit.getRank}")
            val booster = SXGBoost.train(trainingSet, xgBoostConfMap, round, obj = obj, eval = eval)
            Rabit.shutdown()
            logger.info("Stopping XGBoost worker " + id)

            // FIXME: figure out why Booster doesn't serialize properly
            val bos = new ByteArrayOutputStream()
            booster.saveModel(bos)
            booster.getModelDump().foreach(println)
            bos.toByteArray
          } else {
            Rabit.shutdown()
            throw new XGBoostError("Empty bundle in training data")
          }
        }
      }
      .filter(_.nonEmpty)
      .reduce((x, y) => x)
      // FIXME: figure out why Booster doesn't serialize properly
      .map(bytes => SXGBoost.loadModel(new ByteArrayInputStream(bytes)))
  }

  def main(args: Array[String]): Unit = {
    runScio
//    runLocal
//    runParallel
  }

  private val path = "/Users/neville/src/gcp/xgboost/demo/data"
  private val paramMap = Map(
    "eta" -> "1", "max_depth" -> "2", "silent" -> "0", "objective" -> "binary:logistic")

  private def runScio: Unit = {
    val p = PipelineOptionsFactory.create()
    p.setRunner(classOf[InProcessPipelineRunner])
    val sc = ScioContext(p)

    val trainingSet = readFile(path + "/agaricus.txt.train")
    val trainingData = sc.parallelize(trainingSet)
    val f = XGBoost.train(trainingData, paramMap, 10, 1).materialize
    sc.close()
    val booster = f.waitForResult().value.next()

    val testMat = new DMatrix(path + "/agaricus.txt.test")
    val result = booster.predict(testMat)
    booster.getModelDump().foreach(println)
    testMat.getLabel.zip(result.map(_.head)).take(20).foreach(println)
    println(testMat.getLabel.zip(result.map(_.head))
      .map(p => math.pow(p._1 - p._2, 2.0)).sum / result.length)
  }

  private def runLocal: Unit = {
    val trainMat = new DMatrix(path + "/agaricus.txt.train")
    val testMat = new DMatrix(path + "/agaricus.txt.test")
    val booster = SXGBoost.train(trainMat, paramMap, 5)

    val result = booster.predict(testMat)
    booster.getModelDump().foreach(println)
    testMat.getLabel.zip(result.map(_.head)).take(20).foreach(println)
    println(testMat.getLabel.zip(result.map(_.head))
      .map(p => math.pow(p._1 - p._2, 2.0)).sum / result.length)
  }

  private def runParallel: Unit = {
    val trainMat = new DMatrix(path + "/agaricus.txt.train")
    val testMat = new DMatrix(path + "/agaricus.txt.test")

    val tracker = new RabitTracker(1)
    tracker.start()
    val env = tracker.getWorkerEnvs.asScala

    var booster: Booster = null
    new Thread() {
      override def run(): Unit = {
        Rabit.init((env + ("DMLC_TASK_ID" -> "0")).asJava)
        booster = SXGBoost.train(trainMat, paramMap, 5)
        Rabit.shutdown()
      }
    }.run()
    println(tracker.waitFor())
    val result = booster.predict(testMat)
    booster.getModelDump().foreach(println)
    testMat.getLabel.zip(result.map(_.head)).take(20).foreach(println)
    println(testMat.getLabel.zip(result.map(_.head))
      .map(p => math.pow(p._1 - p._2, 2.0)).sum / result.length)
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
