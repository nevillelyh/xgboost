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

import java.io.File

import ml.dmlc.xgboost4j.LabeledPoint
import ml.dmlc.xgboost4j.java.{Rabit, RabitTracker}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.io.Source

object DebugMat {

  private val path = System.getProperty("user.home") + "/src/gcp/xgboost/demo/data"
  private val paramMap = Map(
    "eta" -> "1", "max_depth" -> "2", "silent" -> "0", "objective" -> "binary:logistic")

  def main(args: Array[String]): Unit = {
    // This works
    // val trainMat = new DMatrix(path + "/agaricus.txt.train")

    // This does not
    // val trainMat = new DMatrix(readFile(path + "/agaricus.txt.train").iterator)

    val testMat = new DMatrix(path + "/agaricus.txt.test")

    val tracker = new RabitTracker(1)
    tracker.start()
    val env = tracker.getWorkerEnvs.asScala

    var booster: Booster = null
    new Thread() {
      override def run(): Unit = {
        Rabit.init((env + ("DMLC_TASK_ID" -> "0")).asJava)
//        val trainMat = new DMatrix(path + "/agaricus.txt.train")
        val trainMat = new DMatrix(readFile(path + "/agaricus.txt.train").iterator)
        booster = SXGBoost.train(trainMat, paramMap, 10, Map("train" -> trainMat))
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
