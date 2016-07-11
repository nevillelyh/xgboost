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

import java.io.{ByteArrayInputStream, File}

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

  @throws(classOf[XGBoostError])
  def train(trainingData: SCollection[LabeledPoint], configMap: Map[String, Any],
            round: Int, nWorkers: Int, obj: ObjectiveTrait = null, eval: EvalTrait = null,
            useExternalMemory: Boolean = false): SCollection[Booster] = {
    require(nWorkers > 0, "nWorkers must be greater than 0")
    val xgBoostConfMap = configMap + ("nthread" -> 1)

    val sc = trainingData.context
    val rabitEnv = sc.parallelize(Seq(nWorkers))
      .map { n =>
        installPackages()

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
      // .union(sc.parallelize(Seq((nWorkers, LabeledPoint.fromDenseVector(0f, Array(0f))))))
      .groupByKey
      .cross(rabitEnv)
      .map { case ((id, trainingSamples), env) =>
        installPackages()

        // ====================
        // master node
        // ====================

        logger.info("Starting XGBoost tracker")
        val tracker = new RabitTracker(nWorkers)
        require(tracker.start(), "Failed to start tracker")
        val newEnv = tracker.getWorkerEnvs.asScala
        val isMaster = newEnv("DMLC_TRACKER_URI") == env("DMLC_TRACKER_URI") &&
          newEnv("DMLC_TRACKER_PORT") == env("DMLC_TRACKER_PORT")
        if (!isMaster) {
          tracker.stop()
        }

        // ====================
        // worker node
        // ====================

        logger.info("Starting XGBoost worker")
        val initEnv = env + ("DMLC_TASK_ID" -> id.toString)
        Rabit.init(initEnv.asJava)

        val iter = trainingSamples.iterator
        val boosterData = if (iter.hasNext) {
          val cacheFileName: String = if (useExternalMemory) {
            s"xgboost-dtrain-cache-$id"
          } else {
            null
          }
          val trainingSet = makeMatrix(trainingSamples)
          val booster = SXGBoost.train(
            trainingSet, xgBoostConfMap, round,
            Map("train" -> trainingSet),
            obj = obj, eval = eval)
          Rabit.shutdown()
          logger.info("Stopping XGBoost worker " + id)

          // FIXME: figure out why Booster doesn't serialize properly
          booster.toByteArray
        } else {
          Rabit.shutdown()
          throw new XGBoostError("Empty bundle in training data")
        }

        if (isMaster) {
          logger.info("Stopping XGBoost tracker")
          if (tracker.waitFor() != 0) {
            throw new XGBoostError("XGBoostModel training failed")
          }
        }
        boosterData
      }
      .filter(_.nonEmpty)
      .reduce((x, y) => x)
      // FIXME: figure out why Booster doesn't serialize properly
      .map(bytes => SXGBoost.loadModel(new ByteArrayInputStream(bytes)))
  }

  private def installPackages(): Unit = {
    logger.info("Installing Debian packages")
    exec("apt-get update")
    exec("apt-get install -y python libgomp1")
  }

  private def exec(cmd: String): Unit = {
    val p = Runtime.getRuntime.exec(cmd)
    if (p.waitFor() != 0) {
      logger.error(Source.fromInputStream(p.getErrorStream).getLines().mkString("\n"))
      logger.error(Source.fromInputStream(p.getInputStream).getLines().mkString("\n"))
    }
  }

  private def makeMatrix(data: Iterable[LabeledPoint]): DMatrix = {
    val labels = ListBuffer.empty[Float]
    val values = ListBuffer.empty[Array[Float]]
    val iter = data.iterator
    while (iter.hasNext) {
      val p = iter.next()
      labels.append(p.label)
      values.append(p.values)
    }
    // FIXME: figure out why Iterator[LabeledPoint] doesn't work
    val m = new DMatrix(values.toArray.flatten, values.size, values.head.length)
    m.setLabel(labels.toArray)
    m
  }

  def main(args: Array[String]): Unit = {
    runScio(args)
//    runLocal
//    runParallel
  }

  private val path = System.getProperty("user.home") + "/src/gcp/xgboost/demo/data"
  private val paramMap = Map(
    "eta" -> "1", "max_depth" -> "2", "silent" -> "0", "objective" -> "binary:logistic")

  private def runScio(cmdlineArgs: Array[String]): Unit = {
//    val p = PipelineOptionsFactory.create()
//    p.setRunner(classOf[InProcessPipelineRunner])
//    val sc = ScioContext()
//    val trainingSet = readFile(path + "/agaricus.txt.train")
//    val trainingData = sc.parallelize(trainingSet)

    val (sc, args) = ContextAndArgs(cmdlineArgs)
    val trainingData = sc
      .textFile("gs://neville-steel-eu/xgboost/agaricus.txt.train")
      .map(fromSVMStringToLabeledPoint)

    val f = XGBoost.train(trainingData, paramMap, 10, 2).materialize
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
//    val trainMat = new DMatrix(path + "/agaricus.txt.train")
    val trainMat = new DMatrix(readFile(path + "/agaricus.txt.train").iterator)
    val testMat = new DMatrix(path + "/agaricus.txt.test")
    val booster = SXGBoost.train(trainMat, paramMap, 10)

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
        val mat = new DMatrix(readFile(path + "/agaricus.txt.train").iterator)
        booster = SXGBoost.train(mat, paramMap, 10)
//        booster = SXGBoost.train(trainMat, paramMap, 10)
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
